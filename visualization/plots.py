"""
plots.py — Funciones de visualización para métricas y parámetros del modelo.

Todas las funciones DEVUELVEN la figura (no la renderizan).
El renderizado ocurre en app.py con st.plotly_chart() o st.pyplot().
"""

import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte un color hex + alpha en string rgba() para Plotly."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ─────────────────────────────────────────────
# Curvas de métricas de entrenamiento
# ─────────────────────────────────────────────

def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
) -> go.Figure:
    """Curva de loss de entrenamiento y validación por época."""
    epochs = list(range(1, len(train_losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses,
        mode="lines+markers",
        name="Train Loss",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_losses,
        mode="lines+markers",
        name="Val Loss",
        line=dict(color="#DD8452", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="Evolución del Loss",
        xaxis_title="Época",
        yaxis_title="CrossEntropy Loss",
        legend=dict(x=0.75, y=0.95),
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_accuracy_curve(
    train_accs: List[float],
    val_accs: List[float],
) -> go.Figure:
    """Curva de accuracy de entrenamiento y validación por época."""
    epochs = list(range(1, len(train_accs) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_accs,
        mode="lines+markers",
        name="Train Accuracy",
        line=dict(color="#55A868", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_accs,
        mode="lines+markers",
        name="Val Accuracy",
        line=dict(color="#C44E52", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="Evolución de la Accuracy",
        xaxis_title="Época",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(x=0.05, y=0.05),
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )
    return fig


# ─────────────────────────────────────────────
# Visualización de pesos
# ─────────────────────────────────────────────

def plot_weight_heatmap(
    weight_tensor: torch.Tensor,
    layer_name: str = "Capa",
) -> go.Figure:
    """
    Muestra los pesos de una capa como mapa de calor 2D.

    Cada fila = una neurona de salida.
    Cada columna = una conexión de entrada.
    Colores divergentes centrados en 0: azul = peso negativo, rojo = positivo.
    """
    w = weight_tensor.cpu().numpy()

    fig = go.Figure(data=go.Heatmap(
        z=w,
        colorscale="RdBu_r",
        zmid=0,                # Centrar la escala de color en cero
        colorbar=dict(title="Peso"),
    ))
    fig.update_layout(
        title=f"Pesos — {layer_name} ({w.shape[0]} neuronas × {w.shape[1]} entradas)",
        xaxis_title="Índice de entrada",
        yaxis_title="Índice de neurona",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_first_layer_receptive_fields(
    weight_tensor: torch.Tensor,
    n_neurons: int = 16,
) -> plt.Figure:
    """
    Muestra los pesos de la primera capa como imágenes 28×28.

    Cada neurona de la primera capa tiene 784 pesos (uno por píxel de entrada).
    Al reshapearlos a 28×28 vemos QUÉ PATRÓN detecta esa neurona.
    Esto es equivalente a los "filtros" de una CNN, pero para una red densa.
    """
    w = weight_tensor.cpu().numpy()   # shape: (n_neurons_layer1, 784)
    n = min(n_neurons, w.shape[0])    # No mostrar más neuronas de las que hay

    cols = 8
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(
        "Campos receptivos de la 1ª capa oculta\n"
        "(cada imagen = patrón que detecta una neurona)",
        fontsize=11,
    )

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n:
            neuron_weights = w[i].reshape(28, 28)
            # Usar colormap divergente: permite ver pesos positivos y negativos
            vmax = max(abs(neuron_weights.max()), abs(neuron_weights.min()))
            ax.imshow(neuron_weights, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"N{i}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Visualización de bias
# ─────────────────────────────────────────────

def plot_bias_bars(
    bias_tensor: torch.Tensor,
    layer_name: str = "Capa",
) -> go.Figure:
    """
    Muestra el bias de cada neurona en una capa como gráfico de barras.

    El bias determina cuánto necesita activarse una neurona "por defecto".
    Bias positivo → la neurona se activa más fácilmente.
    Bias negativo → la neurona es más selectiva.
    """
    b = bias_tensor.cpu().numpy()
    indices = list(range(len(b)))

    colors = ["#C44E52" if val < 0 else "#4C72B0" for val in b]

    fig = go.Figure(data=go.Bar(
        x=indices,
        y=b.tolist(),
        marker_color=colors,
        name="Bias",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Bias — {layer_name} ({len(b)} neuronas)",
        xaxis_title="Índice de neurona",
        yaxis_title="Valor del bias",
        template="plotly_white",
        height=300,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# Visualización de predicción
# ─────────────────────────────────────────────

def plot_prediction_probabilities(
    probs: np.ndarray,
    predicted_class: int,
) -> go.Figure:
    """
    Barras con la probabilidad de cada dígito (0-9).
    La clase predicha se resalta en verde.
    """
    classes = [str(i) for i in range(10)]
    colors = [
        "#55A868" if i == predicted_class else "#4C72B0"
        for i in range(10)
    ]

    fig = go.Figure(data=go.Bar(
        x=classes,
        y=(probs * 100).tolist(),
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Probabilidades por clase — Predicción: {predicted_class}",
        xaxis_title="Dígito",
        yaxis_title="Probabilidad (%)",
        yaxis=dict(range=[0, 115]),
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# Muestras del dataset
# ─────────────────────────────────────────────

def plot_mnist_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    n: int = 10,
) -> plt.Figure:
    """Muestra n imágenes de MNIST con sus etiquetas. Útil para entender el dataset."""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.5))

    for i in range(n):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(str(labels[i].item()), fontsize=10)
        axes[i].axis("off")

    fig.suptitle("Muestras de MNIST", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Visualización de arquitectura de la red
# ─────────────────────────────────────────────

# Paleta de colores del grafo
_C_INPUT   = "#A8C4E0"   # Nodos de entrada — azul claro
_C_HIDDEN  = "#9BA8B5"   # Nodos ocultos — gris azulado
_C_OUTPUT  = "#E88080"   # Nodos de salida — salmón
_C_PRED    = "#2ECC71"   # Nodo predicho — verde
_C_POS     = "#5C7BC4"   # Peso positivo — azul
_C_NEG     = "#E8923A"   # Peso negativo — naranja


def _sample_indices(actual: int, display: int) -> np.ndarray:
    """Muestrea `display` índices uniformemente del rango [0, actual)."""
    if actual <= display:
        return np.arange(actual)
    return np.round(np.linspace(0, actual - 1, display)).astype(int)


def _node_ys(n: int) -> np.ndarray:
    """Posiciones Y centradas y equiespaciadas en el rango [-1, 1]."""
    return np.linspace(1.0, -1.0, n) if n > 1 else np.array([0.0])


def _activation_opacities(values: np.ndarray, lo: float = 0.20, hi: float = 1.0) -> np.ndarray:
    """Normaliza valores de activación a opacidades en [lo, hi]."""
    v = np.abs(values)
    span = v.max() - v.min()
    if span < 1e-8:
        return np.full(len(v), (lo + hi) / 2)
    return lo + (hi - lo) * (v - v.min()) / span


def plot_network_architecture(
    hidden_layers_config: List[int],
    weights_list: Optional[List[torch.Tensor]] = None,
    activations: Optional[List[np.ndarray]] = None,
    output_probs: Optional[np.ndarray] = None,
    max_neurons_display: int = 12,
    predicted_class: Optional[int] = None,
) -> go.Figure:
    """
    Grafo interactivo de la arquitectura de la red neuronal.

    Conexiones coloreadas por signo del peso (azul = positivo, naranja = negativo).
    Brillo de los nodos proporcional a la magnitud de la activación.
    Cuando se pasa output_probs, se muestran dígito y probabilidad a la derecha
    de cada nodo de salida.

    Parámetros
    ----------
    hidden_layers_config : List[int]
        Neuronas por capa oculta, e.g. [128, 64].
    weights_list : list of Tensor, opcional
        Un tensor por inter-capa (n_hidden + 1 tensores incluyendo la capa de salida).
        Cada tensor tiene shape (out, in) — igual que nn.Linear.weight.
    activations : list of ndarray, opcional
        Activaciones por capa: [input_784, hidden_1, ..., logits_10].
    output_probs : ndarray de shape (10,), opcional
        Probabilidades softmax para mostrar junto a los nodos de salida.
    max_neurons_display : int
        Máximo de nodos a mostrar en capas ocultas (default 12).
    predicted_class : int, opcional
        Índice del dígito predicho; se resalta en verde.
    """
    INPUT_ACTUAL  = 784
    OUTPUT_ACTUAL = 10
    INPUT_DISPLAY = 10  # siempre se muestran 10 nodos de entrada

    all_actual  = [INPUT_ACTUAL] + list(hidden_layers_config) + [OUTPUT_ACTUAL]
    all_display = [INPUT_DISPLAY] + [min(n, max_neurons_display) for n in hidden_layers_config] + [OUTPUT_ACTUAL]
    n_layers = len(all_actual)

    # ── Posiciones X e Y de cada capa ─────────────────────────────────────
    X_START, X_END = 0.15, 0.80   # espacio para caja de entrada y etiquetas de salida
    x_pos   = np.linspace(X_START, X_END, n_layers)
    layer_ys = [_node_ys(d) for d in all_display]
    layer_idx = [_sample_indices(a, d) for a, d in zip(all_actual, all_display)]

    traces: list = []

    # ── Trazos de conexiones (agrupados en 4 cubos por magnitud y signo) ──
    # Cada cubo acumula segmentos separados por None (truco eficiente de Plotly)
    # cubo: (x_list, y_list, color, width, nombre_leyenda)
    _buckets: dict = {
        "sp": ([], [], _hex_to_rgba(_C_POS, 0.85), 2.0),   # strong positive
        "wp": ([], [], _hex_to_rgba(_C_POS, 0.30), 1.0),   # weak positive
        "wn": ([], [], _hex_to_rgba(_C_NEG, 0.30), 1.0),   # weak negative
        "sn": ([], [], _hex_to_rgba(_C_NEG, 0.85), 2.0),   # strong negative
        "nu": ([], [], "rgba(160,160,160,0.18)", 0.8),      # sin pesos (neutro)
    }

    def _add_edge(bx: list, by: list, x1: float, y1: float, x2: float, y2: float) -> None:
        bx.extend([x1, x2, None])
        by.extend([y1, y2, None])

    for li in range(n_layers - 1):
        x1, x2 = x_pos[li], x_pos[li + 1]
        ys1, ys2 = layer_ys[li], layer_ys[li + 1]

        if weights_list is not None and li < len(weights_list):
            W = weights_list[li].cpu().numpy()           # (out_size, in_size)
            si = layer_idx[li]                           # sampled src indices
            di = layer_idx[li + 1]                       # sampled dst indices
            W_sub = W[np.ix_(di, si)]                   # (n_dst, n_src)
            w_abs_max = np.abs(W_sub).max() + 1e-8
            W_norm = W_sub / w_abs_max                  # normalizado a [-1, 1]

            for j, y2 in enumerate(ys2):
                for k, y1 in enumerate(ys1):
                    w = W_norm[j, k]
                    if w >= 0.4:
                        _add_edge(*_buckets["sp"][:2], x1, y1, x2, y2)
                    elif w >= 0.0:
                        _add_edge(*_buckets["wp"][:2], x1, y1, x2, y2)
                    elif w >= -0.4:
                        _add_edge(*_buckets["wn"][:2], x1, y1, x2, y2)
                    else:
                        _add_edge(*_buckets["sn"][:2], x1, y1, x2, y2)
        else:
            for y2 in ys2:
                for y1 in ys1:
                    _add_edge(*_buckets["nu"][:2], x1, y1, x2, y2)

    for bx, by, color, width in _buckets.values():
        if bx:
            traces.append(go.Scatter(
                x=bx, y=by,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="none",
                showlegend=False,
            ))

    # ── Trazos de nodos por capa ───────────────────────────────────────────
    annotations: list = []

    for li in range(n_layers):
        x_layer = x_pos[li]
        ys       = layer_ys[li]
        n_disp   = all_display[li]
        n_act    = all_actual[li]

        if li == 0:
            layer_type, base_color = "input",  _C_INPUT
        elif li == n_layers - 1:
            layer_type, base_color = "output", _C_OUTPUT
        else:
            layer_type, base_color = "hidden", _C_HIDDEN

        # Opacidades basadas en activaciones
        if layer_type == "output" and output_probs is not None:
            # Para la capa de salida usamos las probabilidades directamente
            opacities = _activation_opacities(output_probs, 0.25, 1.0)
        elif activations is not None and li < len(activations):
            sampled = activations[li][layer_idx[li]]
            opacities = _activation_opacities(sampled, 0.25, 1.0)
        else:
            opacities = np.full(n_disp, 0.85)

        # Color por nodo (el predicho en verde)
        node_colors = []
        for j in range(n_disp):
            is_pred = (layer_type == "output" and predicted_class is not None and j == predicted_class)
            alpha = float(opacities[j])
            color = _C_PRED if is_pred else base_color
            node_colors.append(_hex_to_rgba(color, alpha))

        # Texto hover
        hover = []
        for j in range(n_disp):
            gidx = layer_idx[li][j]
            if layer_type == "input":
                hover.append(f"Entrada #{gidx}<br>({n_act} total)")
            elif layer_type == "output":
                prob_str = f"Prob: {output_probs[j] * 100:.2f}%" if output_probs is not None else ""
                hover.append(f"Dígito {j}<br>{prob_str}")
            else:
                act_str = ""
                if activations is not None and li < len(activations):
                    act_str = f"<br>Act: {activations[li][gidx]:.3f}"
                hover.append(f"Neurona {gidx} (capa oculta {li}){act_str}")

        traces.append(go.Scatter(
            x=[x_layer] * n_disp,
            y=ys.tolist(),
            mode="markers",
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(color="#555555", width=1.5),
                symbol="circle",
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ))

        # Etiquetas de probabilidad a la derecha de la capa de salida
        if layer_type == "output" and output_probs is not None:
            for j in range(n_disp):
                is_pred = (predicted_class is not None and j == predicted_class)
                text_color = _C_PRED if is_pred else "#333333"
                check = " ✓" if is_pred else ""
                annotations.append(dict(
                    x=x_layer + 0.055, y=float(ys[j]),
                    text=f"<b>{j}{check}</b>  {output_probs[j] * 100:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color=text_color),
                    xanchor="left",
                    yanchor="middle",
                ))

        # Indicador "···" cuando se truncan nodos
        if n_disp < n_act:
            annotations.append(dict(
                x=x_layer, y=float(ys[-1]) - 0.15,
                text="···",
                showarrow=False,
                font=dict(size=20, color="#999999"),
                xanchor="center",
            ))

        # Etiqueta de capa debajo
        if li == 0:
            lname = "Input Layer"
        elif li == n_layers - 1:
            lname = "Output Layer"
        else:
            lname = f"Hidden Layer {li}"

        sub = f"({n_act} neuronas)" if n_disp == n_act else f"({n_disp}/{n_act} mostradas)"
        annotations.append(dict(
            x=x_layer, y=-1.20,
            text=f"<b>{lname}</b><br><sub>{sub}</sub>",
            showarrow=False,
            font=dict(size=11),
            xanchor="center",
            yanchor="top",
        ))

    # ── Caja de imagen de entrada ─────────────────────────────────────────
    # Rectángulo blanco a la izquierda del Input Layer con líneas hacia los nodos
    BOX_X0, BOX_X1 = 0.00, 0.10
    BOX_Y0, BOX_Y1 = -0.50, 0.50

    shapes = [dict(
        type="rect",
        x0=BOX_X0, y0=BOX_Y0,
        x1=BOX_X1, y1=BOX_Y1,
        line=dict(color="#888888", width=2),
        fillcolor="rgba(245,245,245,0.6)",
    )]

    # Líneas desde el borde derecho de la caja a los nodos de entrada (escaleadas en Y)
    box_scale = (BOX_Y1 - BOX_Y0) / 2.0   # 0.50
    for y in layer_ys[0]:
        traces.insert(0, go.Scatter(
            x=[BOX_X1, x_pos[0]],
            y=[float(y) * box_scale, float(y)],
            mode="lines",
            line=dict(color="rgba(130,130,130,0.25)", width=0.7),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Trazos de leyenda (invisibles en el gráfico, visibles en el cuadro) ─
    legend_items = [
        (dict(size=12, color=_hex_to_rgba(_C_INPUT, 0.85),  line=dict(color="#555", width=1.5)), "Input node",     "markers"),
        (dict(size=12, color=_hex_to_rgba(_C_HIDDEN, 0.85), line=dict(color="#555", width=1.5)), "Hidden node",    "markers"),
        (dict(size=12, color=_hex_to_rgba(_C_OUTPUT, 0.85), line=dict(color="#555", width=1.5)), "Output node",    "markers"),
    ]
    for marker_kw, name, mode in legend_items:
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode=mode,
            marker=marker_kw,
            name=name,
            showlegend=True,
        ))

    traces.append(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=_hex_to_rgba(_C_POS, 0.85), width=3),
        name="Positive weight",
        showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=_hex_to_rgba(_C_NEG, 0.85), width=3),
        name="Negative weight",
        showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=_hex_to_rgba(_C_PRED, 0.9), line=dict(color="#27AE60", width=1.5)),
        name="Node brightness = activation",
        showlegend=True,
    ))

    # ── Layout final ──────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[-0.05, 1.10]),
        yaxis=dict(visible=False, range=[-1.45, 1.15]),
        template="plotly_white",
        height=580,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            x=0.0, y=-0.02,
            xanchor="left",
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
    )
    return fig


# ─────────────────────────────────────────────
# Saliency map
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    cm: "np.ndarray",
    class_labels: list,
) -> go.Figure:
    """
    Matriz de confusión interactiva normalizada por fila (recall por clase).

    cm : ndarray (n, n) con valores en [0, 1] — fila = real, columna = predicho.
    class_labels : lista de etiquetas de clase ("0"–"9" o "A"–"Z").
    """
    n = len(class_labels)
    pct_text = [[f"{cm[r, c]*100:.1f}%" for c in range(n)] for r in range(n)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        text=pct_text,
        texttemplate="%{text}",
        textfont=dict(size=max(6, 11 - n // 5)),  # smaller font for 26 classes
        colorscale=[
            [0.0,  "#0f1117"],   # dark background = 0%
            [0.15, "#1a3a2a"],
            [0.4,  "#1e7a45"],
            [0.7,  "#27ae60"],
            [1.0,  "#2ECC71"],   # bright green = 100%
        ],
        zmin=0, zmax=1,
        hovertemplate=(
            "<b>Real: %{y}</b><br>"
            "Predicho: %{x}<br>"
            "Porcentaje: %{text}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title="Recall",
            tickformat=".0%",
            thickness=14,
            len=0.8,
        ),
    ))

    # Highlight diagonal with a thin white border overlay
    diag_x = [class_labels[i] for i in range(n)]
    diag_y = [class_labels[i] for i in range(n)]
    fig.add_trace(go.Scatter(
        x=diag_x, y=diag_y,
        mode="markers",
        marker=dict(symbol="square", size=max(6, 22 - n // 2),
                    color="rgba(0,0,0,0)", line=dict(color="white", width=1.5)),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        xaxis=dict(title="Predicho", side="bottom", tickfont=dict(size=max(8, 12 - n // 6))),
        yaxis=dict(title="Real", autorange="reversed",
                   tickfont=dict(size=max(8, 12 - n // 6))),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        margin=dict(l=60, r=20, t=30, b=60),
        height=max(380, 300 + n * 14),
    )
    return fig


def plot_saliency_overlay(
    input_image: np.ndarray,
    saliency: np.ndarray,
    class_label: str,
) -> plt.Figure:
    """
    Muestra el dibujo original, el mapa de saliencia y la superposición en 3 paneles.

    Parámetros
    ----------
    input_image : ndarray (28,28)
        Imagen en rango [0,255] tal como la dibujó el usuario.
    saliency : ndarray (28,28)
        Magnitud del gradiente normalizada a [0,1] — de NeuralNet.compute_saliency().
    class_label : str
        Etiqueta de la clase predicha ("7", "A", etc.) para el título del overlay.
    """
    BG = "#0f1117"
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor=BG)
    fig.patch.set_facecolor(BG)

    titles = ["Tu dibujo", "Importancia de píxeles", f"Overlay  →  predijo: {class_label}"]
    title_colors = ["#aaaaaa", "#aaaaaa", "#2ECC71"]

    # Panel 1 — original drawing
    axes[0].imshow(input_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    # Panel 2 — saliency heatmap (hot: black→red→yellow→white)
    axes[1].imshow(saliency, cmap="hot", vmin=0, vmax=1, interpolation="nearest")

    # Panel 3 — overlay: drawing as dark base + saliency as glowing heatmap
    axes[2].imshow(input_image, cmap="gray", vmin=0, vmax=255, alpha=0.35, interpolation="nearest")
    axes[2].imshow(saliency, cmap="hot", vmin=0, vmax=1, alpha=0.75, interpolation="nearest")

    for ax, title, color in zip(axes, titles, title_colors):
        ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.8)
    return fig


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    class_labels: List[str],
) -> go.Figure:
    """
    Scatter t-SNE interactivo con Plotly, un color por clase.

    coords : ndarray (n, 2)  — coordenadas 2D de t-SNE.
    labels : ndarray (n,)    — índices de clase (enteros).
    class_labels : list       — ["0"…"9"] o ["A"…"Z"].
    """
    # Paleta: 10 colores para dígitos, 26 para letras (ciclamos si hace falta)
    PALETTE = [
        "#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#2980B9", "#8E44AD", "#27AE60",
        "#D35400", "#C0392B", "#16A085", "#7F8C8D", "#2C3E50",
        "#F1C40F", "#95A5A6", "#6C5CE7", "#00B894", "#FD79A8",
        "#FDCB6E", "#55EFC4", "#74B9FF", "#A29BFE", "#FD7272",
        "#B2BEC3",
    ]

    fig = go.Figure()

    n_classes = len(class_labels)
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        if not mask.any():
            continue
        color = PALETTE[cls_idx % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=class_labels[cls_idx],
            marker=dict(
                color=color,
                size=5,
                opacity=0.75,
                line=dict(width=0),
            ),
            hovertemplate=f"<b>{class_labels[cls_idx]}</b><extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="t-SNE — activaciones de la última capa oculta",
            font=dict(color="#FAFAFA", size=14),
            x=0.5,
        ),
        xaxis=dict(title="t-SNE 1", showgrid=False, zeroline=False,
                   tickfont=dict(color="#888"), title_font=dict(color="#888")),
        yaxis=dict(title="t-SNE 2", showgrid=False, zeroline=False,
                   tickfont=dict(color="#888"), title_font=dict(color="#888")),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#FAFAFA"),
        legend=dict(
            title=dict(text="Clase", font=dict(color="#aaa")),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#333",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=520,
    )
    return fig

