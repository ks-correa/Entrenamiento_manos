import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ["mano_abierta", "puno", "paz", "pulgar_arriba"]
DISPLAY_CLASS_NAMES = {
    "mano_abierta": "palma",
    "puno": "puno",
    "paz": "paz",
    "pulgar_arriba": "pulgar_arriba",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
DATASET_LIMITS = {"small": 25, "medium": 50}


class HandPoseDataset(Dataset):
    """Dataset simple basado en rutas de imagen y etiquetas numericas."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def verify_dataset_structure(dataset_dir):
    """Verifica que existan las cuatro carpetas esperadas del proyecto."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {dataset_path}")

    missing_classes = [
        class_name for class_name in CLASS_NAMES if not (dataset_path / class_name).is_dir()
    ]
    if missing_classes:
        missing = ", ".join(missing_classes)
        raise FileNotFoundError(f"Faltan estas carpetas de clases: {missing}")

    return dataset_path


def find_images(class_dir):
    """Devuelve las imagenes validas dentro de una carpeta de clase."""
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_dataset_samples(dataset_dir, dataset_size, seed=42, min_images_per_class=None):
    """
    Carga rutas de imagen usando maximo 25 o 50 imagenes por clase.

    Si aun no hay suficientes imagenes para una clase, usa las disponibles y
    muestra un aviso. Esto permite avanzar mientras se completa el dataset.
    """
    if dataset_size not in DATASET_LIMITS:
        valid_sizes = ", ".join(DATASET_LIMITS)
        raise ValueError(f"dataset_size debe ser uno de: {valid_sizes}")

    dataset_path = verify_dataset_structure(dataset_dir)
    max_per_class = DATASET_LIMITS[dataset_size]
    if min_images_per_class is not None:
        max_per_class = max(max_per_class, min_images_per_class)
    rng = random.Random(seed)
    samples_by_class = {}

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_path / class_name
        images = find_images(class_dir)

        if not images:
            raise ValueError(f"La clase '{class_name}' no tiene imagenes.")

        rng.shuffle(images)
        selected_images = images[:max_per_class]

        if min_images_per_class is None and len(images) < max_per_class:
            print(
                f"Aviso: clase '{class_name}' tiene {len(images)} imagenes; "
                f"se usaran todas para '{dataset_size}'."
            )

        samples_by_class[class_name] = [(image_path, label) for image_path in selected_images]

    return samples_by_class


def split_samples(
    samples_by_class,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    train_images_per_class=None,
):
    """Divide cada clase en entrenamiento, validacion y prueba."""
    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    test_samples = []

    for class_name, samples in samples_by_class.items():
        class_samples = list(samples)
        rng.shuffle(class_samples)
        total = len(class_samples)

        if total < 3:
            raise ValueError(
                f"La clase '{class_name}' necesita al menos 3 imagenes para dividir "
                "en train/val/test."
            )

        if train_images_per_class is not None:
            if train_images_per_class < 1:
                raise ValueError("train_images_per_class debe ser mayor que 0.")
            if total < train_images_per_class + 2:
                raise ValueError(
                    f"La clase '{class_name}' necesita al menos "
                    f"{train_images_per_class + 2} imagenes: "
                    f"{train_images_per_class} para entrenamiento, 1 para validacion "
                    "y 1 para prueba."
                )

            train_count = train_images_per_class
            remaining = total - train_count
            val_count = max(1, int(remaining / 2))
        else:
            train_count = max(1, int(total * train_ratio))
            val_count = max(1, int(total * val_ratio))

        if train_count + val_count >= total:
            val_count = 1
            train_count = total - 2

        train_samples.extend(class_samples[:train_count])
        val_samples.extend(class_samples[train_count : train_count + val_count])
        test_samples.extend(class_samples[train_count + val_count :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def get_train_transforms():
    """Transformaciones con data augmentation solo para entrenamiento."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms():
    """Transformaciones basicas para validacion y prueba."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    dataset_dir,
    dataset_size,
    batch_size,
    seed=42,
    num_workers=0,
    train_images_per_class=None,
):
    """Crea DataLoaders de PyTorch para train, val y test."""
    min_images_per_class = None
    if train_images_per_class is not None:
        min_images_per_class = train_images_per_class + 2

    samples_by_class = load_dataset_samples(
        dataset_dir,
        dataset_size,
        seed=seed,
        min_images_per_class=min_images_per_class,
    )
    train_samples, val_samples, test_samples = split_samples(
        samples_by_class,
        seed=seed,
        train_images_per_class=train_images_per_class,
    )

    train_dataset = HandPoseDataset(train_samples, transform=get_train_transforms())
    val_dataset = HandPoseDataset(val_samples, transform=get_eval_transforms())
    test_dataset = HandPoseDataset(test_samples, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print_dataset_summary(samples_by_class, train_samples, val_samples, test_samples)

    return train_loader, val_loader, test_loader, CLASS_NAMES


def print_dataset_summary(samples_by_class, train_samples, val_samples, test_samples):
    """Muestra un resumen para confirmar que se usa el dataset propio."""
    train_counts = count_samples_by_class(train_samples)
    val_counts = count_samples_by_class(val_samples)
    test_counts = count_samples_by_class(test_samples)

    print("\nResumen del dataset:")
    for class_name, samples in samples_by_class.items():
        print(
            f"  {class_name}: {len(samples)} imagenes seleccionadas | "
            f"train: {train_counts[class_name]} | "
            f"val: {val_counts[class_name]} | "
            f"test: {test_counts[class_name]}"
        )
    print(f"  Entrenamiento: {len(train_samples)} imagenes")
    print(f"  Validacion: {len(val_samples)} imagenes")
    print(f"  Prueba: {len(test_samples)} imagenes\n")


def count_samples_by_class(samples):
    """Cuenta cuantas muestras hay de cada clase en una lista."""
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1
    return counts


def get_device():
    """Selecciona GPU si esta disponible; de lo contrario usa CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(outputs, labels):
    """Calcula accuracy de un lote."""
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def save_training_plot(history, output_path, title):
    """Guarda graficas de loss y accuracy de entrenamiento/validacion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Val accuracy")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_metrics_csv(rows, output_path):
    """Guarda metricas en CSV usando pandas."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def append_metrics_csv(row, output_path):
    """Agrega una fila de metricas a un CSV sin borrar evaluaciones anteriores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
