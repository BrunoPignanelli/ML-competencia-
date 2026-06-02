"""
loader.py — Descarga y preparación de datasets MNIST y EMNIST Letters.

MNIST: 60,000 imágenes de dígitos manuscritos (0-9), 28×28 px.
EMNIST Letters: ~124,800 imágenes de letras manuscritas (A-Z), 28×28 px.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# ── Constantes MNIST ─────────────────────────────────────────────────────────
MNIST_MEAN: float = 0.1307
MNIST_STD:  float = 0.3081
DATA_DIR:   str   = "./mnist_data"

# ── Constantes EMNIST Letters ─────────────────────────────────────────────────
# EMNIST letters labels son 1-indexed (1=A … 26=Z); se corrigen a 0-indexed.
EMNIST_MEAN: float = 0.1722
EMNIST_STD:  float = 0.3309
EMNIST_DATA_DIR: str = "./emnist_data"


class _ZeroIndexedWrapper(Dataset):
    """Wrapper que resta 1 a los labels de EMNIST Letters (1-indexed → 0-indexed)."""

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, label = self._ds[idx]
        return img, label - 1


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


def load_emnist_letters(
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Descarga EMNIST Letters y devuelve DataLoaders de train, validación y test.

    Retorna DataLoaders con 26 clases (A=0 … Z=25).
    La primera descarga requiere ~50 MB.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
    ])

    # EMNIST labels son 1-indexed; el wrapper los convierte a 0-indexed
    full_train = _ZeroIndexedWrapper(
        datasets.EMNIST(
            root=EMNIST_DATA_DIR,
            split="letters",
            train=True,
            download=True,
            transform=transform,
        )
    )
    test_dataset = _ZeroIndexedWrapper(
        datasets.EMNIST(
            root=EMNIST_DATA_DIR,
            split="letters",
            train=False,
            download=True,
            transform=transform,
        )
    )

    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator  = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
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
