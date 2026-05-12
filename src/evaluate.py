import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from train_cnn import CustomCNN
from train_resnet import build_resnet18
from utils import CLASS_NAMES, append_metrics_csv, create_dataloaders, get_device


def load_model(model_path, architecture, num_classes, device):
    """Carga un modelo guardado segun la arquitectura indicada."""
    if architecture == "cnn":
        model = CustomCNN(num_classes=num_classes)
    elif architecture == "resnet":
        model = build_resnet18(num_classes=num_classes)
    else:
        raise ValueError("architecture debe ser 'cnn' o 'resnet'.")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """Obtiene predicciones y etiquetas reales del conjunto de prueba."""
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions


def save_confusion_matrix(matrix, class_names, output_path, title):
    """Guarda una matriz de confusion como imagen."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, matrix[row, col], ha="center", va="center", color=color)

    plt.ylabel("Etiqueta real")
    plt.xlabel("Prediccion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelos guardados de poses de mano.")
    parser.add_argument("--model", required=True, help="Ruta al archivo .pth del modelo.")
    parser.add_argument("--architecture", choices=["cnn", "resnet"], required=True)
    parser.add_argument("--dataset", default="dataset", help="Ruta al dataset propio.")
    parser.add_argument("--dataset_size", choices=["small", "medium"], required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Dispositivo usado: {device}")

    _, _, test_loader, class_names = create_dataloaders(
        args.dataset,
        args.dataset_size,
        args.batch_size,
    )

    model = load_model(args.model, args.architecture, len(class_names), device)
    labels, predictions = evaluate_model(model, test_loader, device)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_stem = Path(args.model).stem

    matrix_path = results_dir / f"{model_stem}_confusion_matrix.png"
    metrics_path = results_dir / "evaluation_metrics.csv"

    save_confusion_matrix(
        matrix,
        class_names,
        matrix_path,
        f"Matriz de confusion - {model_stem}",
    )

    append_metrics_csv(
        {
            "model": str(args.model),
            "architecture": args.architecture,
            "dataset_size": args.dataset_size,
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        },
        metrics_path,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision weighted: {precision:.4f}")
    print(f"Recall weighted: {recall:.4f}")
    print(f"F1 weighted: {f1:.4f}")
    print(f"Matriz de confusion guardada en: {matrix_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
