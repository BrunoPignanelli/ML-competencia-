"""
trainer.py — Loop de entrenamiento y validación de la red neuronal.

Este módulo es independiente de Streamlit: no importa st ni conoce la UI.
La comunicación con la UI se hace a través de callbacks opcionales.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.neural_net import NeuralNet


# Tipo del callback que se llama al final de cada época
# Firma: (epoch: int, metrics: dict) → None
EpochCallback = Callable[[int, Dict[str, float]], None]


class Trainer:
    """
    Encapsula el loop de entrenamiento y validación.

    Ejemplo de uso:
        trainer = Trainer(model, lr=0.001)
        history = trainer.fit(train_loader, val_loader, epochs=10)
    """

    def __init__(
        self,
        model: NeuralNet,
        lr: float = 0.001,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.lr = lr

        # Selección automática de dispositivo (GPU si está disponible)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Adam: optimizer adaptativo, robusto ante distintos learning rates
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # CrossEntropyLoss espera logits (no softmax) y etiquetas enteras
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Ejecuta una época de entrenamiento completa.

        Retorna
        -------
        avg_loss : float
            Loss promedio sobre todos los batches.
        accuracy : float
            Porcentaje de predicciones correctas (0-100).
        """
        self.model.train()  # Activa dropout y batch norm si existen
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Paso forward
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Paso backward
            self.optimizer.zero_grad()  # Limpiar gradientes del batch anterior
            loss.backward()             # Calcular gradientes
            self.optimizer.step()       # Actualizar pesos

            # Acumular métricas
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evalúa el modelo sobre un DataLoader sin actualizar pesos.

        Retorna
        -------
        avg_loss : float
        accuracy : float
        """
        self.model.eval()  # Desactiva dropout durante evaluación
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        epoch_callback: Optional[EpochCallback] = None,
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo por N épocas y devuelve el historial de métricas.

        Parámetros
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
        epoch_callback : callable, opcional
            Se llama al final de cada época con (epoch_num, metrics_dict).
            Úsalo para actualizar la UI de Streamlit sin bloquear.

        Retorna
        -------
        history : dict con listas "train_loss", "train_acc", "val_loss", "val_acc"
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Notificar a la UI (sin acoplar este módulo a Streamlit)
            if epoch_callback is not None:
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                epoch_callback(epoch, metrics)

        return history

    @torch.no_grad()
    def confusion_matrix(self, loader: DataLoader, n_classes: int) -> "np.ndarray":
        """
        Calcula la matriz de confusión normalizada por fila sobre un DataLoader.

        Retorna un array (n_classes, n_classes) donde cada fila suma 1.0.
        Fila = clase real, columna = clase predicha.
        """
        import numpy as np
        self.model.eval()
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)

        for images, labels in loader:
            images  = images.to(self.device)
            labels  = labels.to(self.device)
            preds   = self.model(images).argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[true, pred] += 1

        # Normalize by row (true label) → recall per class; avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        return cm / np.where(row_sums == 0, 1, row_sums)
