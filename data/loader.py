"""
loader.py — Descarga y preparación del dataset MNIST.

MNIST tiene 60,000 imágenes de entrenamiento y 10,000 de test.
Cada imagen es un dígito manuscrito de 28×28 píxeles en escala de grises.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Valores estándar de normalización para MNIST
# Calculados sobre el conjunto de entrenamiento completo
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081

# Directorio local donde se guardará el dataset descargado
DATA_DIR: str = "./mnist_data"


def get_transforms() -> transforms.Compose:
    """Devuelve el pipeline de transformaciones para MNIST."""
    return transforms.Compose([
        transforms.ToTensor(),
        # Normalizar: (pixel - mean) / std → valores aprox. en [-1, 1]
        # Esto acelera la convergencia del entrenamiento
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def load_mnist(
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Descarga MNIST y devuelve DataLoaders de train, validación y test.

    Parámetros
    ----------
    batch_size : int
        Tamaño del mini-batch para entrenamiento.
    val_split : float
        Fracción del set de entrenamiento usada para validación (0.0-1.0).
    num_workers : int
        Hilos para carga paralela de datos (0 = sin paralelismo, más compatible).

    Retorna
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    transform = get_transforms()

    # Descarga automática si no existe localmente
    full_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    # Dividir entrenamiento en train + validación
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    # random_split usa un generator para reproducibilidad
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,         # Mezclar en cada época es crucial para SGD
        num_workers=num_workers,
        pin_memory=False,
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

    return train_loader, val_loader, test_loader


def get_sample_images(
    n: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Devuelve n imágenes del set de test sin normalizar (para visualización).
    Útil para mostrar ejemplos reales de dígitos MNIST en la UI.
    """
    raw_transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=raw_transform,
    )
    images, labels = zip(*[dataset[i] for i in range(n)])
    return torch.stack(images), torch.tensor(labels)
