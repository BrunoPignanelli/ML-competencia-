"""
app.py — Aplicación educativa MNIST.

Diseño centrado en el estudiante:
  - Sidebar: entrenar el modelo (un clic, defaults sensatos).
  - Pantalla principal: dibujar un dígito → predicción automática instantánea.
  - Tabs secundarios: curvas de entrenamiento y visualización de pesos (opcional).
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from data.loader import load_mnist
from model.neural_net import NeuralNet
from training.trainer import Trainer
from visualization.network_html import render_network_html
from visualization.plots import (
    plot_accuracy_curve,
    plot_bias_bars,
    plot_first_layer_receptive_fields,
    plot_loss_curve,
    plot_prediction_probabilities,
    plot_weight_heatmap,
)

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False


# ── Página ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MNIST — Red Neuronal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS mínimo: agranda el número predicho y pule el canvas
st.markdown("""
<style>
.big-digit {
    font-size: 120px;
    font-weight: 900;
    text-align: center;
    line-height: 1;
    padding: 10px 0;
}
.correct   { color: #2ECC71; }
.uncertain { color: #F39C12; }
.wrong     { color: #E74C3C; }
.status-ok   { background:#d4edda; color:#155724; padding:8px 14px; border-radius:8px; font-weight:600; }
.status-warn { background:#fff3cd; color:#856404; padding:8px 14px; border-radius:8px; font-weight:600; }
section[data-testid="stSidebar"] { min-width: 320px !important; }
</style>
""", unsafe_allow_html=True)


# ── Estado global ─────────────────────────────────────────────────────────────

def _init():
    defaults = dict(
        model=None,
        trained=False,
        test_acc=0.0,
        train_losses=[],
        val_losses=[],
        train_accs=[],
        val_accs=[],
        last_activations=None,   # activaciones de la última predicción
        last_probs=None,         # probabilidades de la última predicción
        last_pred=None,          # clase predicha en la última predicción
        last_input_image=None,   # imagen 28×28 de la última predicción
        show_result=False,       # True tras presionar Predecir
        attempt=0,               # contador para reiniciar el canvas
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Panel de entrenamiento
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 Configuración")
    st.caption("Ajustá los parámetros y entrenà el modelo.")

    # ── Hiperparámetros ───────────────────────────────────────────────────────
    with st.expander("⚙️ Hiperparámetros", expanded=True):
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

    # ── Botón entrenar ────────────────────────────────────────────────────────
    train_btn = st.button("🚀 Entrenar modelo", type="primary", use_container_width=True)

    # ── Estado del modelo ─────────────────────────────────────────────────────
    if st.session_state["trained"]:
        acc = st.session_state["test_acc"]
        color = "#2ECC71" if acc >= 95 else "#F39C12"
        st.markdown(
            f"<div class='status-ok'>✅ Modelo entrenado &nbsp;|&nbsp; "
            f"<span style='color:{color}'><b>{acc:.1f}%</b></span> accuracy en test</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='status-warn'>⏳ Modelo no entrenado aún</div>",
                    unsafe_allow_html=True)

    # ── Lógica de entrenamiento (dentro del sidebar para que los placeholders queden acá)
    if train_btn:
        st.session_state["train_losses"] = []
        st.session_state["val_losses"] = []
        st.session_state["train_accs"] = []
        st.session_state["val_accs"] = []
        st.session_state["trained"] = False

        prog = st.progress(0, text="Cargando MNIST…")
        status = st.empty()

        with st.spinner("Cargando MNIST…"):
            train_loader, val_loader, test_loader = load_mnist(batch_size=batch_size)

        model = NeuralNet(hidden_layers=hidden_layers)
        st.session_state["model"] = model
        trainer = Trainer(model=model, lr=lr)

        # Placeholders de métricas en vivo
        col_a, col_b = st.columns(2)
        ph_loss = col_a.empty()
        ph_acc  = col_b.empty()
        ph_charts = st.empty()

        def on_epoch(epoch, metrics):
            pct = epoch / epochs
            prog.progress(pct, text=f"Época {epoch}/{epochs} — val acc {metrics['val_acc']:.1f}%")
            ph_loss.metric("Val Loss", f"{metrics['val_loss']:.4f}")
            ph_acc.metric("Val Acc",   f"{metrics['val_acc']:.1f}%")
            st.session_state["train_losses"].append(metrics["train_loss"])
            st.session_state["val_losses"].append(metrics["val_loss"])
            st.session_state["train_accs"].append(metrics["train_acc"])
            st.session_state["val_accs"].append(metrics["val_acc"])

        trainer.fit(train_loader, val_loader, epochs=epochs, epoch_callback=on_epoch)

        _, test_acc = trainer.evaluate(test_loader)
        st.session_state["test_acc"] = test_acc
        st.session_state["trained"] = True
        prog.progress(1.0, text=f"✅ Listo — {test_acc:.1f}% en test")
        st.rerun()   # refrescar para mostrar badge verde y habilitar canvas

    # ── Consejos rápidos ──────────────────────────────────────────────────────
    with st.expander("💡 Guía rápida"):
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
    "✏️ Dibujá y predecí",
    "📈 Curvas de entrenamiento",
    "🔬 Pesos del modelo",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Canvas → Predecir → Red animada → Resultado
# ════════════════════════════════════════════════════════════════════════════

with tab_draw:

    if not st.session_state["trained"]:
        st.info("👈 Entrenà el modelo desde el panel izquierdo para empezar.")
        st.stop()

    if not CANVAS_OK:
        st.error("Instalá `streamlit-drawable-canvas`: `pip install streamlit-drawable-canvas`")
        st.stop()

    # ── FASE 1: dibujo + botón Predecir ──────────────────────────────────────
    if not st.session_state["show_result"]:

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("### ✏️ Dibujá un dígito")
            st.caption("Trazos blancos sobre fondo negro. Presioná 🗑️ para borrar.")

            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#1a1a1a",
                width=320,
                height=320,
                drawing_mode="freedraw",
                update_streamlit=True,
                key=f"canvas_{st.session_state['attempt']}",
            )

        with col_right:
            st.markdown("### 🔮 Cuando estés listo...")
            st.caption("Asegurate de que el número sea grande y centrado.")

            img_data    = canvas_result.image_data if canvas_result is not None else None
            has_drawing = img_data is not None and img_data[:, :, :3].sum() > 0

            st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)

            predict_btn = st.button(
                "🔮 Predecir",
                type="primary",
                use_container_width=True,
                disabled=not has_drawing,
                help="Dibujá un número primero" if not has_drawing else "¡A predecir!",
            )

        # ── Lógica de predicción ────────────────────────────────────────────
        if predict_btn and has_drawing:
            gray     = img_data[:, :, 0].astype(np.float32)
            pil_img  = Image.fromarray(gray.astype(np.uint8))
            pil_img  = pil_img.resize((28, 28), Image.LANCZOS)
            img_28   = np.array(pil_img, dtype=np.float32)
            img_norm = (img_28 / 255.0 - 0.1307) / 0.3081
            tensor   = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

            net: NeuralNet = st.session_state["model"]
            net.eval()
            probs = net.predict_proba(tensor).squeeze().numpy()
            pred  = int(probs.argmax())
            _, acts = net.forward_with_activations(tensor)

            st.session_state["last_activations"] = acts
            st.session_state["last_probs"]        = probs
            st.session_state["last_pred"]         = pred
            st.session_state["last_input_image"]  = img_28
            st.session_state["show_result"]       = True
            st.rerun()

    # ── FASE 2: red animada + resultado ──────────────────────────────────────
    else:
        net: NeuralNet = st.session_state["model"]
        pred  = st.session_state["last_pred"]
        probs = st.session_state["last_probs"]
        conf  = float(probs[pred]) * 100

        st.markdown("### 🧠 La red neuronal en acción")
        st.caption("La red se construye capa a capa. Luego podés hacer hover sobre cualquier nodo.")

        with st.expander("💡 ¿Cómo leer este grafo?"):
            st.markdown("""
- **Nodos azul** = entrada (10 píxeles representativos de 784). El recuadro es el dígito dibujado.
- **Nodos grises** = capas ocultas (ReLU). El **brillo y halo** reflejan la magnitud de activación.
- **Nodo verde** = dígito predicho — los **3 pulsos** marcan el resultado.
- **Líneas azules** = pesos positivos · **Naranja** = pesos negativos. Grosor ∝ magnitud del peso.
- **Partículas** = señal fluyendo en tiempo real tras la construcción de la red.
- **Hover** sobre cualquier nodo para ver activación exacta y estado ReLU.
            """)

        html_str = render_network_html(
            hidden_layers_config=net.hidden_layers_config,
            weights_list=net.get_all_weights(),
            activations=st.session_state.get("last_activations"),
            output_probs=probs,
            max_neurons_display=12,
            predicted_class=pred,
            input_image=st.session_state.get("last_input_image"),
            height=530,
        )
        components.html(html_str, height=535, scrolling=False)

        # ── Resultado ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🎯 Resultado")

        res_col, btn_col = st.columns([3, 1], gap="large")

        with res_col:
            bar_color = "#2ECC71" if conf >= 80 else "#F39C12" if conf >= 50 else "#E74C3C"
            msg = (
                "Alta confianza ✅" if conf >= 80
                else "Confianza media ⚠️" if conf >= 50
                else "Confianza baja ❌ — intentá dibujar más grande y centrado"
            )
            css_class = "correct" if conf >= 80 else "uncertain" if conf >= 50 else "wrong"

            st.markdown(
                f"<div class='big-digit {css_class}'>{pred}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
<div style="margin:6px 0 20px 0;">
  <div style="background:#f0f0f0; border-radius:8px; height:18px; overflow:hidden;">
    <div style="width:{conf:.0f}%; background:{bar_color}; height:100%; border-radius:8px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:4px;
              font-size:13px; color:#555;">
    <span>{msg}</span><span><b>{conf:.1f}%</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("**Probabilidad por dígito:**")
            rows_html = ""
            for i, p in enumerate(probs):
                pct   = p * 100
                bold  = "font-weight:900; font-size:15px;" if i == pred else "font-size:14px;"
                color = bar_color if i == pred else "#4C72B0"
                rows_html += f"""
<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
  <span style="width:14px; text-align:right; {bold}">{i}</span>
  <div style="flex:1; background:#f0f0f0; border-radius:6px; height:20px; overflow:hidden;">
    <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:6px;"></div>
  </div>
  <span style="width:48px; font-size:13px; color:#333;">{pct:.1f}%</span>
</div>"""
            st.markdown(f"<div style='padding:4px 0'>{rows_html}</div>", unsafe_allow_html=True)

        with btn_col:
            st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Intentar de nuevo", use_container_width=True):
                st.session_state["show_result"]      = False
                st.session_state["attempt"]          += 1
                st.session_state["last_activations"] = None
                st.session_state["last_probs"]       = None
                st.session_state["last_pred"]        = None
                st.session_state["last_input_image"] = None
                st.rerun()

            with st.expander("🔬 28×28"):
                img_28 = st.session_state.get("last_input_image")
                if img_28 is not None:
                    st.image(img_28 / 255.0, width=112, caption="Lo que ve la red")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Curvas de entrenamiento
# ════════════════════════════════════════════════════════════════════════════

with tab_curves:
    if not st.session_state["trained"]:
        st.info("Entrenà el modelo primero.")
    else:
        acc = st.session_state["test_acc"]
        n_ep = len(st.session_state["train_losses"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy en test",   f"{acc:.2f}%")
        c2.metric("Épocas entrenadas",  n_ep)
        c3.metric("Mejor val accuracy", f"{max(st.session_state['val_accs']):.2f}%")

        col_l, col_a = st.columns(2)
        with col_l:
            st.plotly_chart(
                plot_loss_curve(st.session_state["train_losses"],
                                st.session_state["val_losses"]),
                use_container_width=True,
            )
        with col_a:
            st.plotly_chart(
                plot_accuracy_curve(st.session_state["train_accs"],
                                    st.session_state["val_accs"]),
                use_container_width=True,
            )

        with st.expander("💡 ¿Cómo leer estas curvas?"):
            st.markdown("""
- **Loss (pérdida):** queremos que baje. Si la val loss sube mientras train loss baja → **overfitting**.
- **Accuracy:** queremos que suba. Para MNIST, 5 épocas con config default → ~97–98%.
- La **brecha** entre entrenamiento y validación indica si la red memoriza o generaliza.
            """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pesos del modelo
# ════════════════════════════════════════════════════════════════════════════

with tab_weights:
    if not st.session_state["trained"]:
        st.info("Entrenà el modelo primero.")
    else:
        net: NeuralNet = st.session_state["model"]

        # ── Campos receptivos primera capa ────────────────────────────────────
        st.subheader("Campos receptivos — 1ª capa oculta")
        with st.expander("💡 ¿Qué son?"):
            st.markdown("""
Cada neurona de la 1ª capa tiene **784 pesos** (uno por píxel 28×28).
Al visualizarlos como imagen vemos **qué patrón activa esa neurona**.
Rojo = pesos positivos (se activa con esos píxeles). Azul = pesos negativos.
Después de entrenar vas a ver formas que recuerdan trazos de dígitos.
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

