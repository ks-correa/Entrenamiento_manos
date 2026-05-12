import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from utils import calculate_accuracy, create_dataloaders, get_device, save_metrics_csv, save_training_plot


def build_resnet18(num_classes=4):
    """Carga ResNet18 preentrenada y reemplaza la ultima capa."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # Se congelan las capas base para usar ImageNet como punto de partida.
    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in tqdm(dataloader, desc="Entrenando", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += calculate_accuracy(outputs, labels) * images.size(0)

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validando", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_acc += calculate_accuracy(outputs, labels) * images.size(0)

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size


def main():
    parser = argparse.ArgumentParser(description="Entrenar ResNet18 para poses de mano.")
    parser.add_argument("--dataset", default="dataset", help="Ruta al dataset propio.")
    parser.add_argument("--dataset_size", choices=["small", "medium"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    device = get_device()
    print(f"Dispositivo usado: {device}")

    train_loader, val_loader, _, class_names = create_dataloaders(
        args.dataset,
        args.dataset_size,
        args.batch_size,
    )

    model = build_resnet18(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    metrics_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoca {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

    model_path = Path("models") / f"resnet_{args.dataset_size}.pth"
    torch.save(
        {
            "architecture": "resnet",
            "dataset_size": args.dataset_size,
            "class_names": class_names,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )

    plot_path = Path("results") / f"resnet_{args.dataset_size}_training.png"
    metrics_path = Path("results") / f"resnet_{args.dataset_size}_training_metrics.csv"
    save_training_plot(history, plot_path, f"ResNet18 - dataset {args.dataset_size}")
    save_metrics_csv(metrics_rows, metrics_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Grafica guardada en: {plot_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
