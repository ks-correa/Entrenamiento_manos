import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from predict import (
    display_name,
    load_trained_model,
    save_prediction_visualization,
    sorted_prediction_percentages,
)
from utils import CLASS_NAMES, IMAGE_EXTENSIONS, get_device, get_eval_transforms


def find_input_images(folder):
    """Busca imagenes dentro de una carpeta, incluyendo subcarpetas."""
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_true_label(image_path, input_folder):
    """Intenta inferir la etiqueta real usando el nombre de la subcarpeta."""
    relative_parent = image_path.parent.relative_to(input_folder)
    if not relative_parent.parts:
        return None

    folder_name = relative_parent.parts[0]
    if folder_name in CLASS_NAMES:
        return folder_name
    return None


UNCERTAIN_LABEL = "no_seguro"


def predict_images(model, image_paths, input_folder, class_names, device, confidence_threshold):
    """Clasifica una lista de imagenes y devuelve filas con resultados."""
    transform = get_eval_transforms()
    rows = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            rows.append(
                {
                    "image": str(image_path),
                    "true_label": "",
                    "raw_predicted_label": "",
                    "predicted_label": "",
                    "confidence": "",
                    "is_uncertain": "",
                    "error": str(error),
                }
            )
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)
            confidence, predicted_index = torch.max(probabilities, dim=0)

        predicted_class = class_names[predicted_index.item()]
        confidence_value = confidence.item()
        is_uncertain = confidence_value < confidence_threshold
        final_label = UNCERTAIN_LABEL if is_uncertain else display_name(predicted_class)
        true_label = infer_true_label(image_path, input_folder)
        row = {
            "image": str(image_path),
            "true_label": display_name(true_label) if true_label else "",
            "raw_predicted_label": display_name(predicted_class),
            "predicted_label": final_label,
            "confidence": confidence_value,
            "confidence_percentage": confidence_value * 100,
            "is_uncertain": is_uncertain,
            "error": "",
        }

        for class_name, probability in zip(class_names, probabilities.cpu().tolist()):
            row[f"prob_{display_name(class_name)}"] = probability
            row[f"prob_pct_{display_name(class_name)}"] = probability * 100

        rows.append(row)

    return rows


def build_summary_rows(valid_rows):
    """Construye un resumen por categoria predicha."""
    total = len(valid_rows)
    summary_rows = []
    rows_by_class = defaultdict(list)

    for row in valid_rows:
        rows_by_class[row["predicted_label"]].append(row)

    for label in sorted(rows_by_class):
        class_rows = rows_by_class[label]
        confidences = [float(row["confidence"]) for row in class_rows]
        count = len(class_rows)
        percentage = count / total if total else 0.0
        summary_rows.append(
            {
                "predicted_label": label,
                "count": count,
                "percentage": percentage,
                "average_confidence": sum(confidences) / count if count else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
            }
        )

    return summary_rows


def row_probabilities(row, class_names):
    """Extrae las probabilidades de una fila en el mismo orden de class_names."""
    return [float(row[f"prob_{display_name(class_name)}"]) for class_name in class_names]


def build_visualization_path(image_path, input_folder, output_dir):
    """Construye un nombre estable para la imagen anotada."""
    relative_path = Path(image_path).relative_to(input_folder)
    stem_parts = relative_path.with_suffix("").parts
    filename = "__".join(stem_parts)
    safe_filename = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in filename
    )
    return Path(output_dir) / f"{safe_filename}_prediction.png"


def save_folder_visualizations(rows, input_folder, output_dir, class_names):
    """Guarda una visualizacion anotada por cada imagen clasificada."""
    output_dir = Path(output_dir)
    saved_paths = []

    for row in rows:
        if row["error"]:
            continue

        image_path = Path(row["image"])
        probabilities = row_probabilities(row, class_names)
        output_path = build_visualization_path(image_path, input_folder, output_dir)

        save_prediction_visualization(
            image_path=image_path,
            output_path=output_path,
            class_names=class_names,
            probabilities=probabilities,
            predicted_label=row["raw_predicted_label"],
            confidence=float(row["confidence"]),
            final_label=row["predicted_label"],
            true_label=row["true_label"] or None,
        )
        saved_paths.append(output_path)

    return saved_paths


def print_summary_table(summary_rows, total):
    """Imprime una tabla compacta con cantidades, porcentajes y confianza."""
    print("\nResumen general:")
    print(f"{'Categoria':<18} {'Cantidad':>8} {'Porcentaje':>11} {'Conf. prom.':>12}")
    print("-" * 53)

    for row in summary_rows:
        print(
            f"{row['predicted_label']:<18} "
            f"{row['count']:>8} "
            f"{row['percentage'] * 100:>10.2f}% "
            f"{row['average_confidence'] * 100:>11.2f}%"
        )

    print("-" * 53)
    print(f"{'Total':<18} {total:>8} {100:>10.2f}%")


def format_probability_line(row, class_names):
    """Formatea los porcentajes por categoria para consola."""
    probabilities = row_probabilities(row, class_names)
    return " | ".join(
        f"{class_name}: {percentage:.2f}%"
        for class_name, percentage in sorted_prediction_percentages(class_names, probabilities)
    )


def print_unlabeled_metrics(valid_rows, confidence_threshold):
    """Imprime metricas disponibles cuando no hay etiquetas reales."""
    total = len(valid_rows)
    confidences = [float(row["confidence"]) for row in valid_rows]
    average_confidence = sum(confidences) / total if total else 0.0
    uncertain_count = sum(row["predicted_label"] == UNCERTAIN_LABEL for row in valid_rows)

    print("\nMetricas sin etiquetas reales:")
    print(f"  Imagenes clasificadas: {total}")
    print(f"  Confianza promedio: {average_confidence * 100:.2f}%")
    print(f"  Umbral de no seguro: {confidence_threshold * 100:.2f}%")
    print(f"  Imagenes marcadas como no_seguro: {uncertain_count}")

    counts = Counter(row["predicted_label"] for row in valid_rows)
    confidence_by_class = defaultdict(list)
    for row in valid_rows:
        confidence_by_class[row["predicted_label"]].append(float(row["confidence"]))

    print("\nPredicciones por clase:")
    for class_name, count in counts.most_common():
        class_confidences = confidence_by_class[class_name]
        class_average = sum(class_confidences) / len(class_confidences)
        print(f"  {class_name}: {count} imagenes | confianza promedio: {class_average * 100:.2f}%")


def print_labeled_metrics(valid_rows, class_names):
    """Imprime metricas reales cuando las imagenes estan en subcarpetas por clase."""
    labels = [row["true_label"] for row in valid_rows]
    predictions = [row["predicted_label"] for row in valid_rows]
    display_class_names = [display_name(class_name) for class_name in class_names] + [UNCERTAIN_LABEL]

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=display_class_names,
        average="weighted",
        zero_division=0,
    )

    print("\nMetricas con etiquetas reales:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision weighted: {precision:.4f}")
    print(f"  Recall weighted: {recall:.4f}")
    print(f"  F1 weighted: {f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=display_class_names,
            zero_division=0,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clasificar todas las imagenes de una carpeta usando un modelo entrenado."
    )
    parser.add_argument("--model", required=True, help="Ruta al archivo .pth entrenado.")
    parser.add_argument("--folder", default="imagen_prueba", help="Carpeta con imagenes a clasificar.")
    parser.add_argument("--architecture", choices=["cnn", "resnet"], default=None)
    parser.add_argument(
        "--visual_output",
        default=None,
        help="Carpeta donde se guardan las imagenes anotadas. Por defecto se guarda en results/.",
    )
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="No genera imagenes anotadas con prediccion y porcentajes.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.60,
        help="Confianza minima para aceptar una clase. Debajo de este valor queda como no_seguro.",
    )
    parser.add_argument(
        "--hide_predictions",
        action="store_true",
        help="Oculta en consola la prediccion individual de cada imagen.",
    )
    args = parser.parse_args()

    input_folder = Path(args.folder)
    model_path = Path(args.model)

    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not input_folder.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {input_folder}")

    image_paths = find_input_images(input_folder)
    if not image_paths:
        valid_extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"No se encontraron imagenes en {input_folder}. Extensiones: {valid_extensions}")

    device = get_device()
    model, class_names, architecture = load_trained_model(model_path, args.architecture, device)
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence_threshold debe estar entre 0 y 1.")

    rows = predict_images(
        model,
        image_paths,
        input_folder,
        class_names,
        device,
        args.confidence_threshold,
    )

    model_stem = model_path.stem
    visual_output_dir = args.visual_output
    if visual_output_dir is None:
        visual_output_dir = Path("results") / f"{model_stem}_{input_folder.name}_visualizations"

    valid_rows = [row for row in rows if not row["error"]]
    error_rows = [row for row in rows if row["error"]]
    summary_rows = build_summary_rows(valid_rows)
    visualization_paths = []
    if not args.no_visualizations:
        visualization_paths = save_folder_visualizations(
            valid_rows,
            input_folder,
            visual_output_dir,
            class_names,
        )

    print(f"Modelo: {model_path}")
    print(f"Arquitectura: {architecture}")
    print(f"Carpeta: {input_folder}")
    print(f"Imagenes encontradas: {len(image_paths)}")
    print(f"Imagenes clasificadas: {len(valid_rows)}")
    print(f"Imagenes con error: {len(error_rows)}")
    print(f"Umbral de no seguro: {args.confidence_threshold * 100:.2f}%")

    print_summary_table(summary_rows, len(valid_rows))

    if not args.hide_predictions:
        print("\nPredicciones:")
        for row in valid_rows:
            print(
                f"  {row['image']} -> {row['predicted_label']} "
                f"(modelo: {row['raw_predicted_label']}, {float(row['confidence']) * 100:.2f}%)"
            )
            print(f"    {format_probability_line(row, class_names)}")

    if error_rows:
        print("\nErrores:")
        for row in error_rows:
            print(f"  {row['image']}: {row['error']}")

    has_true_labels = valid_rows and all(row["true_label"] for row in valid_rows)
    if has_true_labels:
        print_labeled_metrics(valid_rows, class_names)
    else:
        print_unlabeled_metrics(valid_rows, args.confidence_threshold)
        print(
            "\nNota: no se calcula accuracy/precision/recall reales porque "
            "imagen_prueba no esta separada en subcarpetas con etiquetas."
        )

    if not args.no_visualizations:
        print(f"\nImagenes anotadas guardadas en: {visual_output_dir}")
        print(f"Visualizaciones generadas: {len(visualization_paths)}")


if __name__ == "__main__":
    main()
