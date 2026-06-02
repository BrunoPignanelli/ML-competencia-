"""
app.py — Aplicación educativa MNIST.

Diseño centrado en el estudiante:
  - Sidebar: entrenar el modelo (un clic, defaults sensatos).
  - Pantalla principal: dibujar un dígito → predicción automática instantánea.
  - Tabs secundarios: curvas de entrenamiento y visualización de pesos (opcional).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from data.loader import EMNIST_MEAN, EMNIST_STD, load_emnist_letters, load_mnist
from model.neural_net import NeuralNet
from training.trainer import Trainer
from visualization.landing_html import render_landing_html
from visualization.network_html import render_network_html
from visualization.plots import (
    plot_accuracy_curve,
    plot_bias_bars,
    plot_confusion_matrix,
    plot_first_layer_receptive_fields,
    plot_loss_curve,
    plot_prediction_probabilities,
    plot_saliency_overlay,
    plot_tsne,
    plot_weight_heatmap,
)

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False


# ── Sidebar architecture preview ─────────────────────────────────────────────

def _sidebar_preview_svg(hidden_layers: list, output_size: int = 10) -> str:
    """Renders a compact SVG network diagram for the sidebar hyperparameter panel."""
    all_layers  = [784] + hidden_layers + [output_size]
    MAX_NODES   = 9          # max circles per column
    display     = [min(n, MAX_NODES) for n in all_layers]
    N           = len(all_layers)
    W, H        = 260, 140
    col_w       = W / (N + 1)
    NR          = 5          # node radius
    colors      = ["#58a6ff"] + ["#8b949e"] * len(hidden_layers) + ["#3fb950"]

    def node_ys(n: int):
        if n == 1:
            return [H / 2 - 10]
        spacing = min(14, (H - 40) / (n - 1))
        total   = spacing * (n - 1)
        start   = (H - 20 - total) / 2
        return [start + i * spacing for i in range(n)]

    xs      = [(i + 1) * col_w for i in range(N)]
    all_ys  = [node_ys(d) for d in display]

    parts = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{H}" fill="#0f1117" rx="10"/>',
    ]

    # Edges
    for li in range(N - 1):
        for y1 in all_ys[li]:
            for y2 in all_ys[li + 1]:
                parts.append(
                    f'<line x1="{xs[li]:.1f}" y1="{y1:.1f}" '
                    f'x2="{xs[li+1]:.1f}" y2="{y2:.1f}" '
                    f'stroke="#334" stroke-width="0.6"/>'
                )

    # Nodes + labels
    for li in range(N):
        c = colors[li]
        for y in all_ys[li]:
            parts.append(
                f'<circle cx="{xs[li]:.1f}" cy="{y:.1f}" r="{NR}" '
                f'fill="{c}" opacity="0.92"/>'
            )
        if all_layers[li] > MAX_NODES:
            parts.append(
                f'<text x="{xs[li]:.1f}" y="{H - 22}" text-anchor="middle" '
                f'font-size="9" fill="#666">···</text>'
            )
        label = str(all_layers[li])
        parts.append(
            f'<text x="{xs[li]:.1f}" y="{H - 8}" text-anchor="middle" '
            f'font-size="9" fill="#888" font-family="monospace">{label}</text>'
        )

    parts.append('</svg>')
    return ''.join(parts)


# ── Página ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Decode — ML Interactivo",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Landing page gate — show cover before the app ─────────────────────────────
if st.query_params.get("app"):
    st.session_state["entered"] = True

if not st.session_state.get("entered", False):
    # Hide every piece of Streamlit chrome and make the iframe fill the viewport.
    # The iframe wrapper is made position:fixed so it truly covers the full window
    # regardless of Streamlit's internal padding/margins.
    st.markdown("""
    <style>
    /* Hide chrome */
    #MainMenu, header, footer,
    [data-testid="stSidebar"],
    [data-testid="stStatusWidget"],
    [data-testid="stToolbar"] { display: none !important; }

    /* Remove all padding from the Streamlit page shell */
    html, body { overflow: hidden !important; background: #04050f !important; }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .main > div,
    .block-container {
        padding: 0 !important;
        margin:  0 !important;
        max-width: 100% !important;
        background: #04050f !important;
    }

    /* Make the component iframe wrapper cover the entire viewport */
    [data-testid="stCustomComponentV1"] {
        position: fixed !important;
        top: 0 !important; left: 0 !important;
        width:  100vw !important;
        height: 100vh !important;
        z-index: 9999 !important;
        padding: 0 !important;
        margin:  0 !important;
    }
    [data-testid="stCustomComponentV1"] iframe {
        width:  100% !important;
        height: 100% !important;
        border: none !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # height here doesn't matter visually (overridden by CSS above) but must be > 0
    components.html(render_landing_html(), height=900, scrolling=True)
    st.stop()

# ── Professional dark theme ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global dark surface ────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
    min-width: 320px !important;
}
[data-testid="stSidebar"] > div:first-child { background: #161b22 !important; }

/* ── Typography ────────────────────────────────────────────────── */
h1, h2, h3, h4 { color: #e6edf3 !important; }
p, label, .stCaption, .stMarkdown, span { color: #c9d1d9; }

/* ── Block container + columns ─────────────────────────────────── */
.block-container { background: #0d1117 !important; }
[data-testid="column"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: #0d1117 !important;
}
/* Catch any remaining white iframe backgrounds from canvas widget */
.stCustomComponentV1 iframe { background: transparent !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-bottom: 1px solid #21262d !important;
    border-radius: 8px 8px 0 0;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-weight: 500;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: rgba(88,166,255,0.06) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0d1117 !important;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: 1px solid #30363d !important;
    background: #21262d !important;
    color: #c9d1d9 !important;
    transition: background 0.15s, border-color 0.15s;
}
.stButton > button:hover {
    background: #30363d !important;
    border-color: #8b949e !important;
}
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    border-color: #2ea043 !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
}

/* ── Sliders ───────────────────────────────────────────────────── */
[data-testid="stSlider"] {
    color: #c9d1d9 !important;
}

/* ── Expanders ─────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
details summary span { color: #c9d1d9 !important; }

/* ── Metrics ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    padding: 14px 16px !important;
    border-radius: 8px !important;
    border: 1px solid #21262d !important;
}
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* ── Selectbox / Radio ─────────────────────────────────────────── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}
[data-testid="stRadio"] label { color: #c9d1d9 !important; }

/* ── Dividers ──────────────────────────────────────────────────── */
hr { border-color: #21262d !important; }

/* ── Info / warning boxes ──────────────────────────────────────── */
[data-testid="stAlert"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

/* ── Custom classes ────────────────────────────────────────────── */
.big-digit {
    font-size: 110px;
    font-weight: 800;
    text-align: center;
    line-height: 1;
    padding: 10px 0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
.correct   { color: #3fb950; }
.uncertain { color: #d29922; }
.wrong     { color: #f85149; }
.status-ok {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(63,185,80,0.25);
    font-size: 14px;
}
.status-warn {
    background: rgba(210,153,34,0.12);
    color: #d29922;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(210,153,34,0.25);
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Estado global ─────────────────────────────────────────────────────────────

def _init():
    defaults = dict(
        # ── Dígitos (MNIST) ───────────────────────────────────────────────────
        model=None,
        trainer=None,
        trained=False,
        test_acc=0.0,
        train_losses=[],
        val_losses=[],
        train_accs=[],
        val_accs=[],
        last_activations=None,
        last_probs=None,
        last_pred=None,
        last_input_image=None,
        show_result=False,
        attempt=0,
        last_saliency=None,          # mapa de saliencia (28×28) de la última predicción
        confusion_matrix=None,       # matriz de confusión normalizada (10×10)
        tsne_coords=None,            # coordenadas t-SNE (n, 2)
        tsne_labels=None,            # etiquetas de clase para t-SNE (n,)
        # ── Letras (EMNIST) ───────────────────────────────────────────────────
        model_letters=None,
        trainer_letters=None,
        trained_letters=False,
        test_acc_letters=0.0,
        train_losses_letters=[],
        val_losses_letters=[],
        train_accs_letters=[],
        val_accs_letters=[],
        last_activations_letters=None,
        last_probs_letters=None,
        last_pred_letters=None,
        last_input_image_letters=None,
        show_result_letters=False,
        attempt_letters=0,
        last_saliency_letters=None,  # mapa de saliencia para letras
        confusion_matrix_letters=None,  # matriz de confusión normalizada (26×26)
        tsne_coords_letters=None,       # coordenadas t-SNE para letras (n, 2)
        tsne_labels_letters=None,       # etiquetas de clase para t-SNE letras (n,)
        # ── Modo activo ───────────────────────────────────────────────────────
        mode="digits",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Panel de entrenamiento
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Configuracion")
    st.caption("Ajusta los parametros y entrena el modelo.")

    # ── Selector de módulo ────────────────────────────────────────────────────
    st.radio(
        "Módulo",
        options=["digits", "letters"],
        format_func=lambda m: "Digitos (MNIST)" if m == "digits" else "Letras (EMNIST)",
        horizontal=True,
        key="mode",
    )
    mode = st.session_state["mode"]

    # ── Hiperparámetros ───────────────────────────────────────────────────────
    with st.expander("Hiperparametros", expanded=True):
        n_hidden = st.slider("Capas ocultas", 1, 4, 2)

        hidden_layers = []
        for i in range(n_hidden):
            default_n = max(16, 128 // (2**i))   # 128, 64, 32, 16 …
            n = st.slider(f"Neuronas — capa {i+1}", 16, 256, default_n, step=16,
                          key=f"n_{i}")
            hidden_layers.append(n)

        lr = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
        )
        epochs = st.slider("Épocas", 1, 20, 5)
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=64)

    # ── Vista previa de arquitectura ──────────────────────────────────────────
    _out_sz = 26 if mode == "letters" else 10
    param_count = sum(
        a * b + b
        for a, b in zip([784] + hidden_layers, hidden_layers + [_out_sz])
    )
    st.markdown("**Vista previa de la red:**")
    st.markdown(_sidebar_preview_svg(hidden_layers, output_size=_out_sz), unsafe_allow_html=True)
    st.caption(
        f"**{len(hidden_layers) + 2}** capas · "
        f"**{sum(hidden_layers):,}** neuronas ocultas · "
        f"**{param_count:,}** parámetros"
    )

    # ── Botón entrenar ────────────────────────────────────────────────────────
    RETRAIN_EPOCHS = 30
    sfx = "_letters" if mode == "letters" else ""
    out_sz = 26 if mode == "letters" else 10

    existing_net = st.session_state.get(f"model{sfx}")
    same_arch = (
        st.session_state[f"trained{sfx}"]
        and existing_net is not None
        and existing_net.hidden_layers_config == hidden_layers
    )

    btn_label = f"Seguir entrenando ({RETRAIN_EPOCHS} epocas)" if same_arch else "Entrenar modelo"
    train_btn = st.button(btn_label, type="primary", use_container_width=True)

    # ── Estado del modelo ─────────────────────────────────────────────────────
    if st.session_state[f"trained{sfx}"]:
        acc = st.session_state[f"test_acc{sfx}"]
        color = "#2ECC71" if acc >= 95 else "#F39C12"
        st.markdown(
            f"<div class='status-ok'>Modelo entrenado &nbsp;|&nbsp; "
            f"<span style='color:{color}'><b>{acc:.1f}%</b></span> accuracy en test</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='status-warn'>Modelo no entrenado</div>",
                    unsafe_allow_html=True)

    # ── Lógica de entrenamiento ───────────────────────────────────────────────
    if train_btn:
        if mode == "letters":
            prog = st.progress(0, text="Descargando EMNIST Letters (~50 MB, solo la primera vez)…")
            with st.spinner("Cargando EMNIST Letters…"):
                train_loader, val_loader, test_loader = load_emnist_letters(batch_size=batch_size)
        else:
            prog = st.progress(0, text="Cargando MNIST…")
            with st.spinner("Cargando MNIST…"):
                train_loader, val_loader, test_loader = load_mnist(batch_size=batch_size)

        if same_arch:
            # Continue from where we left off — Adam optimizer momentum carries over
            model   = st.session_state[f"model{sfx}"]
            trainer = st.session_state[f"trainer{sfx}"]
            epochs_to_run = RETRAIN_EPOCHS
        else:
            # Fresh model for new architecture or first training
            model   = NeuralNet(hidden_layers=hidden_layers, output_size=out_sz)
            trainer = Trainer(model=model, lr=lr)
            st.session_state[f"model{sfx}"]              = model
            st.session_state[f"trainer{sfx}"]            = trainer
            epochs_to_run = epochs
            st.session_state[f"train_losses{sfx}"]       = []
            st.session_state[f"val_losses{sfx}"]         = []
            st.session_state[f"train_accs{sfx}"]         = []
            st.session_state[f"val_accs{sfx}"]           = []

        st.session_state[f"trained{sfx}"] = False

        col_a, col_b = st.columns(2)
        ph_loss = col_a.empty()
        ph_acc  = col_b.empty()
        ph_live_chart = st.empty()   # actualizado cada época con la curva de loss

        def on_epoch(epoch, metrics):
            pct = epoch / epochs_to_run
            prog.progress(pct, text=f"Época {epoch}/{epochs_to_run} — val acc {metrics['val_acc']:.1f}%")
            ph_loss.metric("Val Loss", f"{metrics['val_loss']:.4f}")
            ph_acc.metric("Val Acc",   f"{metrics['val_acc']:.1f}%")
            st.session_state[f"train_losses{sfx}"].append(metrics["train_loss"])
            st.session_state[f"val_losses{sfx}"].append(metrics["val_loss"])
            st.session_state[f"train_accs{sfx}"].append(metrics["train_acc"])
            st.session_state[f"val_accs{sfx}"].append(metrics["val_acc"])
            # Live chart: mostrar desde la época 2 para que el gráfico tenga al menos 2 puntos
            if epoch >= 2:
                ph_live_chart.plotly_chart(
                    plot_loss_curve(
                        st.session_state[f"train_losses{sfx}"],
                        st.session_state[f"val_losses{sfx}"],
                    ),
                    use_container_width=True,
                )

        trainer.fit(train_loader, val_loader, epochs=epochs_to_run, epoch_callback=on_epoch)

        _, test_acc = trainer.evaluate(test_loader)
        st.session_state[f"test_acc{sfx}"]              = test_acc
        st.session_state[f"trained{sfx}"]               = True
        # Compute confusion matrix on test set
        cm = trainer.confusion_matrix(test_loader, n_classes=out_sz)
        st.session_state[f"confusion_matrix{sfx}"]      = cm
        # Clear stale prediction so activations don't mismatch a new model
        st.session_state[f"show_result{sfx}"]           = False
        st.session_state[f"last_activations{sfx}"]      = None
        st.session_state[f"last_probs{sfx}"]            = None
        st.session_state[f"last_pred{sfx}"]             = None
        st.session_state[f"last_input_image{sfx}"]      = None
        st.session_state[f"last_saliency{sfx}"]         = None
        st.session_state[f"tsne_coords{sfx}"]           = None
        st.session_state[f"tsne_labels{sfx}"]           = None
        prog.progress(1.0, text=f"Completado — {test_acc:.1f}% en test")
        st.rerun()

    # ── Consejos rápidos ──────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("← Volver al inicio", help="Volver a la pagina de bienvenida"):
        st.session_state["entered"] = False
        st.query_params.clear()
        st.rerun()

    with st.expander("Guia rapida"):
        st.markdown("""
**¿Qué hacen los hiperparámetros?**

| Parámetro | Efecto |
|---|---|
| Capas ocultas | Más capas = red más "profunda" |
| Neuronas | Más neuronas = más capacidad |
| Learning rate | Velocidad de aprendizaje |
| Épocas | Veces que ve el dataset completo |
| Batch size | Imágenes por paso de actualización |

**Tips para dibujar:**
- Dibujá el número **grande y centrado**
- Usá trazos **gruesos**
- Si la confianza es baja (<60%), intentá de nuevo
        """)


# ════════════════════════════════════════════════════════════════════════════
# ZONA PRINCIPAL — Tabs
# ════════════════════════════════════════════════════════════════════════════

tab_draw, tab_curves, tab_weights = st.tabs([
    "Dibujo y prediccion",
    "Curvas de entrenamiento",
    "Pesos del modelo",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Canvas → Predecir → Red animada → Resultado
# ════════════════════════════════════════════════════════════════════════════

with tab_draw:

    _mode = st.session_state["mode"]
    _sfx  = "_letters" if _mode == "letters" else ""

    if _mode == "letters":
        _norm_mean, _norm_std = EMNIST_MEAN, EMNIST_STD
        _out_sz      = 26
        _class_labels = [chr(65 + i) for i in range(26)]
        _draw_prompt  = "### Dibuja una letra"
        _draw_caption = "Traza una mayuscula grande y centrada."
        _help_hint    = "Dibuja una letra primero"
        _prob_label   = "**Probabilidad por letra:**"
    else:
        _norm_mean, _norm_std = 0.1307, 0.3081
        _out_sz      = 10
        _class_labels = [str(i) for i in range(10)]
        _draw_prompt  = "### Dibuja un digito"
        _draw_caption = "Trazos blancos sobre fondo negro."
        _help_hint    = "Dibuja un numero primero"
        _prob_label   = "**Probabilidad por digito:**"

    if not st.session_state[f"trained{_sfx}"]:
        st.info("Entrena el modelo desde el panel izquierdo para empezar.")
        st.stop()

    if not CANVAS_OK:
        st.error("Instala `streamlit-drawable-canvas`: `pip install streamlit-drawable-canvas`")
        st.stop()

    # ── FASE 1: dibujo + botón Predecir ──────────────────────────────────────
    if not st.session_state[f"show_result{_sfx}"]:

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown(_draw_prompt)
            st.caption(_draw_caption)

            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#1a1a1a",
                width=320,
                height=320,
                drawing_mode="freedraw",
                update_streamlit=True,
                key=f"canvas_{_mode}_{st.session_state[f'attempt{_sfx}']}",
            )

        with col_right:
            st.markdown("### Cuando estes listo")
            st.caption("Asegurate de que el trazo sea grande y centrado.")

            img_data    = canvas_result.image_data if canvas_result is not None else None
            has_drawing = img_data is not None and img_data[:, :, :3].sum() > 0

            st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)

            predict_btn = st.button(
                "Predecir",
                type="primary",
                use_container_width=True,
                disabled=not has_drawing,
                help=_help_hint if not has_drawing else "Ejecutar prediccion",
            )

        # ── Lógica de predicción ────────────────────────────────────────────
        if predict_btn and has_drawing:
            gray     = img_data[:, :, 0].astype(np.float32)
            pil_img  = Image.fromarray(gray.astype(np.uint8))
            pil_img  = pil_img.resize((28, 28), Image.LANCZOS)
            img_28   = np.array(pil_img, dtype=np.float32)
            img_norm = (img_28 / 255.0 - _norm_mean) / _norm_std
            tensor   = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

            net: NeuralNet = st.session_state[f"model{_sfx}"]
            net.eval()
            probs = net.predict_proba(tensor).squeeze().numpy()
            pred  = int(probs.argmax())
            _, acts = net.forward_with_activations(tensor)

            saliency = net.compute_saliency(tensor, pred)

            st.session_state[f"last_activations{_sfx}"] = acts
            st.session_state[f"last_probs{_sfx}"]        = probs
            st.session_state[f"last_pred{_sfx}"]         = pred
            st.session_state[f"last_input_image{_sfx}"]  = img_28
            st.session_state[f"last_saliency{_sfx}"]     = saliency
            st.session_state[f"show_result{_sfx}"]       = True
            st.rerun()

    # ── FASE 2: red animada + resultado ──────────────────────────────────────
    else:
        net: NeuralNet = st.session_state[f"model{_sfx}"]
        pred  = st.session_state[f"last_pred{_sfx}"]
        probs = st.session_state[f"last_probs{_sfx}"]
        conf  = float(probs[pred]) * 100

        # Display character: letter or digit
        display_char = chr(65 + pred) if _mode == "letters" else str(pred)

        st.markdown("### Red neuronal en accion")
        st.caption("La red se construye capa a capa. Hover sobre cualquier nodo para ver detalles.")

        with st.expander("Como leer este grafo"):
            st.markdown("""
- **Nodos azul** = entrada (10 pixeles representativos de 784). El recuadro es lo dibujado.
- **Nodos grises** = capas ocultas (ReLU). El **brillo y halo** reflejan la magnitud de activacion.
- **Nodo verde** = clase predicha — los **3 pulsos** marcan el resultado.
- **Lineas azules** = pesos positivos · **Naranja** = pesos negativos. Grosor = magnitud del peso.
- **Particulas** = senal fluyendo en tiempo real tras la construccion de la red.
- **Hover** sobre cualquier nodo para ver activacion exacta y estado ReLU.
            """)

        html_str = render_network_html(
            hidden_layers_config=net.hidden_layers_config,
            weights_list=net.get_all_weights(),
            activations=st.session_state.get(f"last_activations{_sfx}"),
            output_probs=probs,
            max_neurons_display=12,
            predicted_class=pred,
            input_image=st.session_state.get(f"last_input_image{_sfx}"),
            height=530,
            output_size=_out_sz,
            class_labels=_class_labels,
        )
        components.html(html_str, height=535, scrolling=False)

        # ── Mapa de saliencia ────────────────────────────────────────────────
        saliency = st.session_state.get(f"last_saliency{_sfx}")
        img_28   = st.session_state.get(f"last_input_image{_sfx}")
        if saliency is not None and img_28 is not None:
            with st.expander("Mapa de saliencia — en que se fijo la red", expanded=True):
                st.caption(
                    "Los píxeles **blancos/amarillos** influyeron más en la predicción. "
                    "Los **negros/rojos oscuros** fueron ignorados."
                )
                fig_sal = plot_saliency_overlay(img_28, saliency, display_char)
                st.pyplot(fig_sal, use_container_width=True)
                plt.close(fig_sal)

        # ── Resultado ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Resultado")

        res_col, btn_col = st.columns([3, 1], gap="large")

        with res_col:
            bar_color = "#3fb950" if conf >= 80 else "#d29922" if conf >= 50 else "#f85149"
            msg = (
                "Alta confianza" if conf >= 80
                else "Confianza media" if conf >= 50
                else "Confianza baja — intenta dibujar mas grande y centrado"
            )
            css_class = "correct" if conf >= 80 else "uncertain" if conf >= 50 else "wrong"

            st.markdown(
                f"<div class='big-digit {css_class}'>{display_char}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
<div style="margin:6px 0 20px 0;">
  <div style="background:#21262d; border-radius:8px; height:18px; overflow:hidden;">
    <div style="width:{conf:.0f}%; background:{bar_color}; height:100%; border-radius:8px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:4px;
              font-size:13px; color:#8b949e;">
    <span>{msg}</span><span><b>{conf:.1f}%</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown(_prob_label)
            rows_html = ""
            for i, p in enumerate(probs):
                pct   = p * 100
                bold  = "font-weight:900; font-size:15px;" if i == pred else "font-size:14px;"
                color = bar_color if i == pred else "#58a6ff"
                lbl   = _class_labels[i]
                rows_html += f"""
<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
  <span style="width:18px; text-align:right; color:#c9d1d9; {bold}">{lbl}</span>
  <div style="flex:1; background:#21262d; border-radius:6px; height:20px; overflow:hidden;">
    <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:6px;"></div>
  </div>
  <span style="width:48px; font-size:13px; color:#8b949e;">{pct:.1f}%</span>
</div>"""
            st.markdown(f"<div style='padding:4px 0'>{rows_html}</div>", unsafe_allow_html=True)

        with btn_col:
            st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
            if st.button("Intentar de nuevo", use_container_width=True):
                st.session_state[f"show_result{_sfx}"]      = False
                st.session_state[f"attempt{_sfx}"]          += 1
                st.session_state[f"last_activations{_sfx}"] = None
                st.session_state[f"last_probs{_sfx}"]       = None
                st.session_state[f"last_pred{_sfx}"]        = None
                st.session_state[f"last_input_image{_sfx}"] = None
                st.session_state[f"last_saliency{_sfx}"]    = None
                st.rerun()

            with st.expander("Input 28x28"):
                img_28 = st.session_state.get(f"last_input_image{_sfx}")
                if img_28 is not None:
                    st.image(img_28 / 255.0, width=112, caption="Lo que ve la red")

        # ── t-SNE en Tab 1 ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Embedding space — donde cae tu dibujo")
        st.caption(
            "El t-SNE muestra cómo la red agrupa internamente todas las clases. "
            f"La estrella marca el cluster de **{display_char}**, la clase predicha."
        )

        with st.expander("Que es el t-SNE"):
            st.markdown("""
Cada punto representa una imagen del test set vista a traves de la **ultima capa oculta** de la red.
t-SNE las comprime a 2D conservando la estructura de vecindad.

- **Clusters bien separados** — la red aprendio representaciones discriminativas.
- **Clusters solapados** — esas clases se parecen para la red (ej. 4 y 9, o I y J).
- La estrella indica el centroide del cluster de la clase predicha.
            """)

        _tsne_n1 = st.slider("Muestras", 200, 2000, 1000, step=100, key=f"tsne_n1_{_sfx}")

        if st.button("Generar t-SNE", key=f"tsne_btn1_{_sfx}"):
            with st.spinner("Extrayendo activaciones y calculando t-SNE…"):
                from sklearn.manifold import TSNE as _TSNE

                if _mode == "letters":
                    _, _, _tsne_loader1 = load_emnist_letters(batch_size=256)
                else:
                    _, _, _tsne_loader1 = load_mnist(batch_size=256)

                _embeds1, _lbls1 = net.extract_embeddings(_tsne_loader1, n_samples=_tsne_n1)
                _coords1 = _TSNE(
                    n_components=2, random_state=42,
                    perplexity=min(30, _tsne_n1 // 10),
                ).fit_transform(_embeds1)

                st.session_state[f"tsne_coords{_sfx}"] = _coords1
                st.session_state[f"tsne_labels{_sfx}"] = _lbls1

        _tc = st.session_state.get(f"tsne_coords{_sfx}")
        _tl = st.session_state.get(f"tsne_labels{_sfx}")
        if _tc is not None and _tl is not None:
            st.plotly_chart(
                plot_tsne(_tc, _tl, _class_labels, highlight_class=pred),
                use_container_width=True,
            )
        else:
            st.info("Presiona **Generar t-SNE** para visualizar el embedding space de la red.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Curvas de entrenamiento
# ════════════════════════════════════════════════════════════════════════════

with tab_curves:
    _sfx2 = "_letters" if st.session_state["mode"] == "letters" else ""
    if not st.session_state[f"trained{_sfx2}"]:
        st.info("Entrenà el modelo primero.")
    else:
        acc = st.session_state[f"test_acc{_sfx2}"]
        n_ep = len(st.session_state[f"train_losses{_sfx2}"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy en test",   f"{acc:.2f}%")
        c2.metric("Épocas entrenadas",  n_ep)
        c3.metric("Mejor val accuracy", f"{max(st.session_state[f'val_accs{_sfx2}']):.2f}%")

        col_l, col_a = st.columns(2)
        with col_l:
            st.plotly_chart(
                plot_loss_curve(st.session_state[f"train_losses{_sfx2}"],
                                st.session_state[f"val_losses{_sfx2}"]),
                use_container_width=True,
            )
        with col_a:
            st.plotly_chart(
                plot_accuracy_curve(st.session_state[f"train_accs{_sfx2}"],
                                    st.session_state[f"val_accs{_sfx2}"]),
                use_container_width=True,
            )

        with st.expander("Como leer estas curvas"):
            st.markdown("""
- **Loss (perdida):** queremos que baje. Si la val loss sube mientras train loss baja — **overfitting**.
- **Accuracy:** queremos que suba. Para MNIST, 5 epocas — ~97-98%. EMNIST Letters — ~85-90%.
- La **brecha** entre entrenamiento y validacion indica si la red memoriza o generaliza.
            """)

        # ── Matriz de confusión ────────────────────────────────────────────────
        cm = st.session_state.get(f"confusion_matrix{_sfx2}")
        if cm is not None:
            st.divider()
            st.subheader("Matriz de confusion (test set)")
            _cm_labels = (
                [chr(65 + i) for i in range(26)]
                if st.session_state["mode"] == "letters"
                else [str(i) for i in range(10)]
            )
            st.plotly_chart(
                plot_confusion_matrix(cm, class_labels=_cm_labels),
                use_container_width=True,
            )
            with st.expander("Como leer la matriz de confusion"):
                st.markdown("""
- **Filas** = clase real. **Columnas** = clase predicha por el modelo.
- Los valores son **porcentajes por fila** (recall por clase). La diagonal perfecta = 100%.
- Un cuadrado brillante **fuera de la diagonal** indica que la red confunde esas dos clases.
- *Ejemplo*: si en la fila "4" el color mas fuerte esta en la columna "9", la red confunde 4s con 9s.
- **Letras frecuentemente confundidas**: C-G, I-J, U-V.
                """)

        # ── t-SNE ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("t-SNE — Representaciones internas de la red")
        with st.expander("Que muestra el t-SNE"):
            st.markdown("""
t-SNE reduce las activaciones de la **última capa oculta** a 2 dimensiones para que podamos verlas.
Cada punto es una imagen del test set; el color indica la clase real.

- **Clusters bien separados** → la red aprendió representaciones discriminativas.
- **Clusters mezclados** → esas clases se parecen para la red (ej. 4 y 9, o I y J).
- Cuanto más entrenás, más separados quedan los clusters.

La computación tarda ~10–20 segundos según la cantidad de muestras.
            """)

        _tsne_n = st.slider("Muestras para t-SNE", 200, 2000, 1000, step=100,
                            key=f"tsne_n_{_sfx2}")

        if st.button("Generar t-SNE", key=f"tsne_btn_{_sfx2}"):
            with st.spinner("Extrayendo activaciones y calculando t-SNE…"):
                from sklearn.manifold import TSNE

                # Recrear el loader con batch_size grande para ser eficientes
                if st.session_state["mode"] == "letters":
                    _, _, _tsne_loader = load_emnist_letters(batch_size=256)
                else:
                    _, _, _tsne_loader = load_mnist(batch_size=256)

                _net: NeuralNet = st.session_state[f"model{_sfx2}"]
                _embeds, _lbls = _net.extract_embeddings(_tsne_loader, n_samples=_tsne_n)

                _tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, _tsne_n // 10))
                _coords = _tsne.fit_transform(_embeds)

                st.session_state[f"tsne_coords{_sfx2}"] = _coords
                st.session_state[f"tsne_labels{_sfx2}"] = _lbls

        _tsne_coords = st.session_state.get(f"tsne_coords{_sfx2}")
        _tsne_labels = st.session_state.get(f"tsne_labels{_sfx2}")
        if _tsne_coords is not None and _tsne_labels is not None:
            _tsne_class_labels = (
                [chr(65 + i) for i in range(26)]
                if st.session_state["mode"] == "letters"
                else [str(i) for i in range(10)]
            )
            st.plotly_chart(
                plot_tsne(_tsne_coords, _tsne_labels, _tsne_class_labels),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pesos del modelo
# ════════════════════════════════════════════════════════════════════════════

with tab_weights:
    _sfx3 = "_letters" if st.session_state["mode"] == "letters" else ""
    if not st.session_state[f"trained{_sfx3}"]:
        st.info("Entrenà el modelo primero.")
    else:
        net: NeuralNet = st.session_state[f"model{_sfx3}"]

        # ── Campos receptivos primera capa ────────────────────────────────────
        st.subheader("Campos receptivos — 1ª capa oculta")
        with st.expander("Que son los campos receptivos"):
            st.markdown("""
Cada neurona de la 1a capa tiene **784 pesos** (uno por pixel 28x28).
Al visualizarlos como imagen vemos **que patron activa esa neurona**.
Rojo = pesos positivos (se activa con esos pixeles). Azul = pesos negativos.
Despues de entrenar vas a ver formas que recuerdan trazos de digitos.
            """)

        n_show = st.slider("Neuronas a mostrar", 4,
                           min(64, net.hidden_layers_config[0]), 16, step=4)
        st.pyplot(plot_first_layer_receptive_fields(net.get_layer_weights(0),
                                                     n_neurons=n_show))

        st.divider()

        # ── Heatmap de pesos y bias por capa ─────────────────────────────────
        st.subheader("Pesos y bias por capa")

        layer_options = [
            f"Capa oculta {i+1}  ({net.hidden_layers_config[i]} neuronas)"
            for i in range(len(net.hidden))
        ]
        sel = st.selectbox("Seleccioná una capa", layer_options)
        idx = layer_options.index(sel)

        cw, cb = st.columns(2)
        with cw:
            st.plotly_chart(
                plot_weight_heatmap(net.get_layer_weights(idx), layer_name=sel),
                use_container_width=True,
            )
        with cb:
            st.plotly_chart(
                plot_bias_bars(net.get_layer_bias(idx), layer_name=sel),
                use_container_width=True,
            )

