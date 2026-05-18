import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import (
    DATASET_LIMITS,
    calculate_accuracy,
    create_dataloaders,
    get_device,
    save_metrics_csv,
    save_training_plot,
)


class CustomCNN(nn.Module):
    """CNN propia para clasificar las cuatro poses de mano."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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
    parser = argparse.ArgumentParser(description="Entrenar CNN propia para poses de mano.")
    parser.add_argument("--dataset", default="dataset", help="Ruta al dataset propio.")
    parser.add_argument("--dataset_size", choices=list(DATASET_LIMITS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--no_balanced_sampling",
        action="store_true",
        help="Desactiva el muestreo balanceado por clase durante entrenamiento.",
    )
    parser.add_argument(
        "--train_images_per_class",
        type=int,
        default=None,
        help="Numero exacto de imagenes por clase que se usaran para entrenamiento.",
    )
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    device = get_device()

    train_loader, val_loader, _, class_names = create_dataloaders(
        args.dataset,
        args.dataset_size,
        args.batch_size,
        train_images_per_class=args.train_images_per_class,
        balanced_sampling=not args.no_balanced_sampling,
    )

    model = CustomCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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

    model_path = Path("models") / f"cnn_{args.dataset_size}.pth"
    torch.save(
        {
            "architecture": "cnn",
            "dataset_size": args.dataset_size,
            "class_names": class_names,
            "balanced_sampling": not args.no_balanced_sampling,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )

    plot_path = Path("results") / f"cnn_{args.dataset_size}_training.png"
    metrics_path = Path("results") / f"cnn_{args.dataset_size}_training_metrics.csv"
    save_training_plot(history, plot_path, f"CNN propia - dataset {args.dataset_size}")
    save_metrics_csv(metrics_rows, metrics_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Grafica guardada en: {plot_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
