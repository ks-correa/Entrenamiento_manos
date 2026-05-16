import argparse
from pathlib import Path

import torch
from PIL import Image

from train_cnn import CustomCNN
from train_resnet import build_resnet18
from utils import CLASS_NAMES, DISPLAY_CLASS_NAMES, get_device, get_eval_transforms


def build_model(architecture, num_classes):
    """Crea la arquitectura indicada para cargar los pesos entrenados."""
    if architecture == "cnn":
        return CustomCNN(num_classes=num_classes)
    if architecture == "resnet":
        return build_resnet18(num_classes=num_classes)
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

    model = build_model(architecture, num_classes=len(class_names))
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


def main():
    parser = argparse.ArgumentParser(description="Predecir la pose de mano en una imagen.")
    parser.add_argument("--model", required=True, help="Ruta al archivo .pth entrenado.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen que se quiere clasificar.")
    parser.add_argument("--architecture", choices=["cnn", "resnet"], default=None)
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

    sorted_results = sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    for class_name, probability in sorted_results:
        print(f"  {display_name(class_name)}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
