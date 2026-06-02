"""
neural_net.py — Definición de la red neuronal feedforward configurable.

La arquitectura se construye dinámicamente según los hiperparámetros elegidos
por el usuario: cantidad de capas ocultas y neuronas por capa.
"""

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# Constantes del problema MNIST
INPUT_SIZE: int = 784   # 28×28 píxeles aplanados
OUTPUT_SIZE: int = 10   # dígitos 0-9


class NeuralNet(nn.Module):
    """
    Red neuronal feedforward configurable para clasificación MNIST.

    Parámetros
    ----------
    hidden_layers : List[int]
        Lista con la cantidad de neuronas en cada capa oculta.
        Ejemplo: [128, 64] crea dos capas ocultas.
    input_size : int
        Dimensión de entrada (default: 784 para MNIST 28×28).
    output_size : int
        Cantidad de clases de salida (default: 10 para dígitos 0-9).
    dropout_rate : float
        Tasa de dropout aplicada entre capas ocultas (0.0 = sin dropout).
    """

    def __init__(
        self,
        hidden_layers: List[int],
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate

        # Construimos las capas dinámicamente con nn.ModuleList
        # para que PyTorch registre correctamente los parámetros
        self.hidden = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for neurons in hidden_layers:
            self.hidden.append(nn.Linear(prev_size, neurons))
            self.dropouts.append(nn.Dropout(p=dropout_rate))
            prev_size = neurons

        # Capa de salida: devuelve logits crudos (sin softmax)
        # CrossEntropyLoss espera logits, no probabilidades
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante con activaciones ReLU entre capas ocultas."""
        # Aplanamos la imagen: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(x.size(0), -1)

        for layer, dropout in zip(self.hidden, self.dropouts):
            x = F.relu(layer(x))   # ReLU introduce no-linealidad
            x = dropout(x)         # Dropout solo activo en modo train

        # La capa de salida NO tiene activación — CrossEntropyLoss la aplica internamente
        return self.output_layer(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Devuelve probabilidades (softmax) para visualización y predicción.
        Usar solo durante inferencia, no en el loop de entrenamiento.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_layer_weights(self, layer_idx: int) -> torch.Tensor:
        """Devuelve los pesos de una capa oculta específica (detached del grafo)."""
        if layer_idx < len(self.hidden):
            return self.hidden[layer_idx].weight.data.clone()
        raise IndexError(f"La red solo tiene {len(self.hidden)} capas ocultas.")

    def get_layer_bias(self, layer_idx: int) -> torch.Tensor:
        """Devuelve el bias de una capa oculta específica (detached del grafo)."""
        if layer_idx < len(self.hidden):
            return self.hidden[layer_idx].bias.data.clone()
        raise IndexError(f"La red solo tiene {len(self.hidden)} capas ocultas.")

    def get_all_weights(self) -> List[torch.Tensor]:
        """Devuelve los pesos de todas las capas en orden: ocultas primero, salida al final."""
        weights = [layer.weight.data.clone() for layer in self.hidden]
        weights.append(self.output_layer.weight.data.clone())
        return weights

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Forward pass que captura las activaciones intermedias para visualización.

        Retorna (logits, activations) donde activations[i] corresponde a:
          - i=0: entrada aplanada (784 valores)
          - i=1..n_hidden: salida de cada capa oculta tras ReLU
          - i=-1: logits de la capa de salida (10 valores)
        """
        with torch.no_grad():
            h = x.view(x.size(0), -1)
            acts: List[np.ndarray] = [h.squeeze(0).cpu().numpy()]

            for layer, dropout in zip(self.hidden, self.dropouts):
                h = F.relu(layer(h))
                acts.append(h.squeeze(0).cpu().numpy())

            logits = self.output_layer(h)
            acts.append(logits.squeeze(0).cpu().numpy())
            return logits, acts

    def compute_saliency(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Calcula el mapa de saliencia: magnitud del gradiente del logit objetivo respecto al input.

        Muestra qué píxeles influyeron más en la predicción de `target_class`.
        Retorna un array (28,28) normalizado a [0,1].
        """
        self.eval()
        # Clone and detach to avoid modifying the original; requires_grad for backprop
        inp = x.clone().detach().requires_grad_(True)
        output = self.forward(inp)
        self.zero_grad()
        output[0, target_class].backward()        # backprop only through the target logit
        saliency = inp.grad.data.abs().squeeze()  # abs: direction irrelevant, magnitude = importance
        sal_np = saliency.cpu().numpy()
        # Gaussian smoothing reduces pixel-level noise, making the heatmap visually cleaner
        sal_np = gaussian_filter(sal_np, sigma=1.0)
        # Normalize to [0,1] for display; +1e-8 avoids division by zero on blank canvas
        sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
        return sal_np

    @torch.no_grad()
    def extract_embeddings(
        self, loader: "torch.utils.data.DataLoader", n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae activaciones de la última capa oculta para n_samples imágenes.

        Retorna (embeddings, labels) donde embeddings tiene shape (n, last_hidden_size).
        Usado para visualización t-SNE.
        """
        self.eval()
        device = next(self.parameters()).device
        all_embeds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        collected = 0

        for images, labels in loader:
            if collected >= n_samples:
                break
            images = images.to(device)
            x = images.view(images.size(0), -1)
            for layer, dropout in zip(self.hidden, self.dropouts):
                x = F.relu(layer(x))
            all_embeds.append(x.cpu().numpy())
            all_labels.append(labels.numpy())
            collected += images.size(0)

        embeddings = np.concatenate(all_embeds)[:n_samples]
        labels_arr = np.concatenate(all_labels)[:n_samples]
        return embeddings, labels_arr

    def count_parameters(self) -> int:
        """Cuenta el total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        """Devuelve un string legible describiendo la arquitectura."""
        lines = [f"Entrada: {self.input_size} neuronas"]
        for i, neurons in enumerate(self.hidden_layers_config):
            lines.append(f"Capa oculta {i+1}: {neurons} neuronas (ReLU)")
        lines.append(f"Salida: {self.output_size} neuronas (logits)")
        lines.append(f"Total parámetros: {self.count_parameters():,}")
        return "\n".join(lines)
