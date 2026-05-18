import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from utils import (
    DATASET_LIMITS,
    calculate_accuracy,
    create_dataloaders,
    get_device,
    save_metrics_csv,
    save_training_plot,
)


def build_resnet18(num_classes=4, classifier_dropout=0.0):
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
    if classifier_dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable_layers(model, fine_tune_layers):
    """Define que partes de ResNet se actualizan durante el entrenamiento."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    if fine_tune_layers == "none":
        return

    if fine_tune_layers == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    layers_by_mode = {
        "layer4": ["layer4"],
        "layer3_layer4": ["layer3", "layer4"],
    }
    for layer_name in layers_by_mode[fine_tune_layers]:
        layer = getattr(model, layer_name)
        for parameter in layer.parameters():
            parameter.requires_grad = True


def count_trainable_parameters(model):
    """Cuenta parametros entrenables para verificar la fase actual."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model, learning_rate, weight_decay):
    """Crea AdamW solo con los parametros que estan activos."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)


def build_criterion(label_smoothing):
    """Crea la funcion de perdida con regularizacion si la version de PyTorch lo permite."""
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing:
            print("Esta version de PyTorch no soporta label_smoothing; se usara CrossEntropyLoss normal.")
        return nn.CrossEntropyLoss()


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
    parser.add_argument("--dataset_size", choices=list(DATASET_LIMITS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--fine_tune_learning_rate", type=float, default=0.00005)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument(
        "--fine_tune_layers",
        choices=["none", "layer4", "layer3_layer4", "all"],
        default="layer4",
        help="Capas de ResNet que se descongelan despues del calentamiento.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
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

    model = build_resnet18(
        num_classes=len(class_names),
        classifier_dropout=args.dropout,
    ).to(device)
    criterion = build_criterion(args.label_smoothing)

    if args.warmup_epochs > 0:
        current_phase = "fc_only"
        set_trainable_layers(model, "none")
        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    else:
        current_phase = "fine_tuning"
        set_trainable_layers(model, args.fine_tune_layers)
        optimizer = build_optimizer(model, args.fine_tune_learning_rate, args.weight_decay)

    print(
        f"Fase inicial: {current_phase} | "
        f"parametros entrenables: {count_trainable_parameters(model):,}"
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    metrics_rows = []
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_acc = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1:
            current_phase = "fine_tuning" if args.fine_tune_layers != "none" else "fc_only"
            set_trainable_layers(model, args.fine_tune_layers)
            optimizer = build_optimizer(model, args.fine_tune_learning_rate, args.weight_decay)
            print(
                f"\nFase nueva: {current_phase} | "
                f"parametros entrenables: {count_trainable_parameters(model):,}"
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        metrics_rows.append(
            {
                "epoch": epoch,
                "phase": current_phase,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss

        print(
            f"Epoca {epoch:02d}/{args.epochs} | "
            f"Fase: {current_phase} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

    model.load_state_dict(best_model_state)
    model_path = Path("models") / f"resnet_{args.dataset_size}.pth"
    torch.save(
        {
            "architecture": "resnet",
            "dataset_size": args.dataset_size,
            "class_names": class_names,
            "classifier_dropout": args.dropout,
            "fine_tune_layers": args.fine_tune_layers,
            "warmup_epochs": args.warmup_epochs,
            "balanced_sampling": not args.no_balanced_sampling,
            "label_smoothing": args.label_smoothing,
            "weight_decay": args.weight_decay,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )

    plot_path = Path("results") / f"resnet_{args.dataset_size}_training.png"
    metrics_path = Path("results") / f"resnet_{args.dataset_size}_training_metrics.csv"
    save_training_plot(history, plot_path, f"ResNet18 - dataset {args.dataset_size}")
    save_metrics_csv(metrics_rows, metrics_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Mejor epoca: {best_epoch} | Val acc: {best_val_acc:.4f} | Val loss: {best_val_loss:.4f}")
    print(f"Grafica guardada en: {plot_path}")
    print(f"Metricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
