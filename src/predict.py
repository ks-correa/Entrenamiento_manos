import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from train_cnn import CustomCNN
from train_resnet import build_resnet18
from utils import CLASS_NAMES, DISPLAY_CLASS_NAMES, get_device, get_eval_transforms


def build_model(architecture, num_classes, classifier_dropout=0.0):
    """Crea la arquitectura indicada para cargar los pesos entrenados."""
    if architecture == "cnn":
        return CustomCNN(num_classes=num_classes)
    if architecture == "resnet":
        return build_resnet18(num_classes=num_classes, classifier_dropout=classifier_dropout)
    raise ValueError("architecture debe ser 'cnn' o 'resnet'.")


def load_trained_model(model_path, architecture, device):
    """Carga un modelo entrenado desde un archivo .pth."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    saved_architecture = checkpoint.get("architecture")

    if architecture is None:
        architecture = saved_architecture

    if architecture is None:
        raise ValueError(
            "No se pudo detectar la arquitectura. Usa --architecture cnn o --architecture resnet."
        )

    classifier_dropout = checkpoint.get("classifier_dropout", 0.0)
    model = build_model(
        architecture,
        num_classes=len(class_names),
        classifier_dropout=classifier_dropout,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, architecture


def predict_image(model, image_path, class_names, device):
    """Predice la clase de una imagen y devuelve las probabilidades."""
    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_class = class_names[predicted_index.item()]
    return predicted_class, confidence.item(), probabilities.cpu().tolist()


def display_name(class_name):
    """Convierte el nombre interno de carpeta al nombre mostrado."""
    return DISPLAY_CLASS_NAMES.get(class_name, class_name)


def sorted_prediction_percentages(class_names, probabilities):
    """Devuelve las categorias ordenadas por probabilidad descendente."""
    return sorted(
        (
            (display_name(class_name), probability * 100)
            for class_name, probability in zip(class_names, probabilities)
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def save_prediction_visualization(
    image_path,
    output_path,
    class_names,
    probabilities,
    predicted_label,
    confidence,
    final_label=None,
    true_label=None,
):
    """Guarda una imagen con la prediccion y porcentajes por categoria."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    labels = [display_name(class_name) for class_name in class_names]
    percentages = [probability * 100 for probability in probabilities]
    predicted_display = display_name(predicted_label)
    shown_label = final_label or predicted_display

    fig, (image_axis, bar_axis) = plt.subplots(
        2,
        1,
        figsize=(8, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    image_axis.imshow(image)
    image_axis.axis("off")
    image_axis.set_title(
        f"Etiqueta: {shown_label} | Mayor porcentaje: {predicted_display} ({confidence * 100:.2f}%)",
        fontsize=12,
        pad=12,
    )
    if true_label:
        image_axis.text(
            0.5,
            -0.06,
            f"Etiqueta real: {true_label}",
            transform=image_axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    colors = ["#2563eb" if label == predicted_display else "#9ca3af" for label in labels]
    bar_axis.barh(labels, percentages, color=colors)
    bar_axis.set_xlim(0, 100)
    bar_axis.set_xlabel("Porcentaje")
    bar_axis.invert_yaxis()

    for index, percentage in enumerate(percentages):
        text_x = min(percentage + 1.5, 96)
        bar_axis.text(text_x, index, f"{percentage:.2f}%", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predecir la pose de mano en una imagen.")
    parser.add_argument("--model", required=True, help="Ruta al archivo .pth entrenado.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen que se quiere clasificar.")
    parser.add_argument("--architecture", choices=["cnn", "resnet"], default=None)
    parser.add_argument(
        "--visual_output",
        default=None,
        help="Ruta del PNG anotado. Por defecto se guarda en results/.",
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="No genera la imagen anotada con prediccion y porcentajes.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    device = get_device()
    model, class_names, architecture = load_trained_model(model_path, args.architecture, device)
    predicted_class, confidence, probabilities = predict_image(
        model,
        image_path,
        class_names,
        device,
    )

    print(f"Modelo: {model_path}")
    print(f"Arquitectura: {architecture}")
    print(f"Imagen: {image_path}")
    print(f"Etiqueta predicha: {display_name(predicted_class)}")
    print(f"Confianza: {confidence * 100:.2f}%")
    print("\nPorcentajes por clase:")

    for class_name, percentage in sorted_prediction_percentages(class_names, probabilities):
        print(f"  {class_name}: {percentage:.2f}%")

    if not args.no_visualization:
        output_path = args.visual_output
        if output_path is None:
            output_path = Path("results") / f"{model_path.stem}_{image_path.stem}_prediction.png"

        save_prediction_visualization(
            image_path=image_path,
            output_path=output_path,
            class_names=class_names,
            probabilities=probabilities,
            predicted_label=predicted_class,
            confidence=confidence,
        )
        print(f"\nImagen anotada guardada en: {output_path}")


if __name__ == "__main__":
    main()
