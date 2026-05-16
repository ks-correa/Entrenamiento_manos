import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from predict import display_name, load_trained_model
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


def predict_images(model, image_paths, input_folder, class_names, device):
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
                    "predicted_label": "",
                    "confidence": "",
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
        true_label = infer_true_label(image_path, input_folder)
        row = {
            "image": str(image_path),
            "true_label": display_name(true_label) if true_label else "",
            "predicted_label": display_name(predicted_class),
            "confidence": confidence.item(),
            "error": "",
        }

        for class_name, probability in zip(class_names, probabilities.cpu().tolist()):
            row[f"prob_{display_name(class_name)}"] = probability

        rows.append(row)

    return rows


def save_predictions_csv(rows, output_path):
    """Guarda las predicciones individuales en CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_unlabeled_metrics(valid_rows):
    """Imprime metricas disponibles cuando no hay etiquetas reales."""
    total = len(valid_rows)
    confidences = [float(row["confidence"]) for row in valid_rows]
    average_confidence = sum(confidences) / total if total else 0.0
    low_confidence_count = sum(confidence < 0.60 for confidence in confidences)

    print("\nMetricas sin etiquetas reales:")
    print(f"  Imagenes clasificadas: {total}")
    print(f"  Confianza promedio: {average_confidence * 100:.2f}%")
    print(f"  Imagenes con confianza menor a 60%: {low_confidence_count}")

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
    display_class_names = [display_name(class_name) for class_name in class_names]

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
        "--output",
        default=None,
        help="Ruta del CSV de salida. Por defecto se guarda en results/.",
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
    rows = predict_images(model, image_paths, input_folder, class_names, device)

    model_stem = model_path.stem
    output_path = args.output
    if output_path is None:
        output_path = Path("results") / f"{model_stem}_imagen_prueba_predictions.csv"

    save_predictions_csv(rows, output_path)

    valid_rows = [row for row in rows if not row["error"]]
    error_rows = [row for row in rows if row["error"]]

    print(f"Modelo: {model_path}")
    print(f"Arquitectura: {architecture}")
    print(f"Carpeta: {input_folder}")
    print(f"Imagenes encontradas: {len(image_paths)}")
    print(f"Imagenes clasificadas: {len(valid_rows)}")
    print(f"Imagenes con error: {len(error_rows)}")

    print("\nPredicciones:")
    for row in valid_rows:
        print(
            f"  {row['image']} -> {row['predicted_label']} "
            f"({float(row['confidence']) * 100:.2f}%)"
        )

    if error_rows:
        print("\nErrores:")
        for row in error_rows:
            print(f"  {row['image']}: {row['error']}")

    has_true_labels = valid_rows and all(row["true_label"] for row in valid_rows)
    if has_true_labels:
        print_labeled_metrics(valid_rows, class_names)
    else:
        print_unlabeled_metrics(valid_rows)
        print(
            "\nNota: no se calcula accuracy/precision/recall reales porque "
            "imagen_prueba no esta separada en subcarpetas con etiquetas."
        )

    print(f"\nCSV guardado en: {output_path}")


if __name__ == "__main__":
    main()
