# CLAUDE.md — MNIST Neural Net Educational App

This file defines all skills, conventions, and guidance for Claude Code to assist on this project at the highest level.

---

## Project Purpose

Build a fully functional, educational, interactive Python application for training and visualizing a neural network that recognizes handwritten digits (MNIST). The goal is pedagogical: users learn *how* training works, not just get predictions.

---

## Repository

- **GitHub**: https://github.com/BrunoPignanelli/ML-competencia-
- **Branches**: `main` (production), `develop` (feature work), `testing` (QA)
- Always create feature branches off `develop`, merge to `main` via PR.

---

## Skills

### SKILL: Architecture Design

- The neural network class (`NeuralNet`) must be **dynamically configurable** at runtime: hidden layer count, neurons per layer, learning rate, batch size, epochs.
- Use `nn.ModuleList` to build layers dynamically — not hardcoded `nn.Sequential`.
- Use `CrossEntropyLoss` (multiclass classification standard).
- Use `Adam` optimizer (adaptive, forgiving for beginners).
- Normalize MNIST with `mean=0.1307, std=0.3081` (standard MNIST normalization).
- Keep model, training logic, visualization, and Streamlit UI in **separate modules**.

### SKILL: File Structure

Every file has a single responsibility:

```
mnist-neural-net-edu/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── app.py                        # Streamlit entry point only
├── model/
│   ├── __init__.py
│   └── neural_net.py             # NeuralNet class (configurable)
├── training/
│   ├── __init__.py
│   └── trainer.py                # Training loop, metrics collection
├── data/
│   ├── __init__.py
│   └── loader.py                 # MNIST download + DataLoader
└── visualization/
    ├── __init__.py
    ├── plots.py                  # Plotly/matplotlib chart functions
    └── network_html.py           # HTML/Canvas animated network component
```

- Never mix training logic into `app.py`.
- Never put visualization logic inside the model.
- `app.py` only calls functions from other modules and manages `st.session_state`.
- `plots.py` handles static Plotly/matplotlib charts (curves, heatmaps, bars).
- `network_html.py` handles the animated 60fps Canvas network visualization.

### SKILL: Streamlit Session State Management

- Store all training results in `st.session_state` so they persist across reruns.
- Key state keys:
  - `st.session_state["model"]` — trained NeuralNet instance
  - `st.session_state["train_losses"]` — list of per-epoch loss
  - `st.session_state["train_accs"]` — list of per-epoch accuracy
  - `st.session_state["val_losses"]` — list of per-epoch val loss
  - `st.session_state["val_accs"]` — list of per-epoch val accuracy
  - `st.session_state["trained"]` — bool flag
  - `st.session_state["last_activations"]` — `List[np.ndarray]` from last prediction
  - `st.session_state["last_probs"]` — `np.ndarray` shape (10,) softmax probs from last prediction
  - `st.session_state["last_pred"]` — `int` predicted class from last prediction
  - `st.session_state["last_input_image"]` — `np.ndarray` (28×28) raw pixel image shown in the 28×28 preview
  - `st.session_state["show_result"]` — `bool` gates the animation + result section in tab_draw
  - `st.session_state["attempt"]` — `int` counter incremented on retry; changing it resets canvas via `key=` trick
- Never use global variables; always read/write via `st.session_state`.
- Use `st.rerun()` carefully — only after state mutations that require UI refresh.

### SKILL: Training Loop Pattern

The trainer must:
1. Accept a `NeuralNet` model, `DataLoader`, optimizer config, and a **callback** function.
2. Call the callback after each epoch with `(epoch, train_loss, train_acc, val_loss, val_acc)`.
3. The callback updates `st.session_state` and calls `st.rerun()` or updates a placeholder — this enables live training visualization.
4. Return `(train_losses, train_accs, val_losses, val_accs)` at the end.
5. Use `torch.no_grad()` for validation passes.
6. Compute accuracy as `correct / total * 100`.

Pattern for live training in Streamlit:
```python
# Use a Streamlit empty placeholder + progress bar
progress_bar = st.progress(0)
status_text = st.empty()
chart_placeholder = st.empty()

for epoch in range(epochs):
    loss, acc = train_one_epoch(model, loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    # update placeholders — no st.rerun() needed in a loop
    progress_bar.progress((epoch + 1) / epochs)
    chart_placeholder.plotly_chart(build_live_chart(...))
```

### SKILL: Dynamic Model Construction

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        # hidden_layers: list of ints, e.g. [128, 64]
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_size
        for size in hidden_layers:
            self.layers.append(nn.Linear(prev, size))
            prev = size
        self.output = nn.Linear(prev, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)  # raw logits for CrossEntropyLoss
```

- Always flatten input in `forward` with `x.view(x.size(0), -1)`.
- Return raw logits (not softmax) — `CrossEntropyLoss` expects logits.
- For prediction with probabilities: apply `F.softmax(logits, dim=1)`.

Key methods on `NeuralNet` beyond `forward`:
- `predict_proba(x)` — returns softmax probabilities under `torch.no_grad()`.
- `get_layer_weights(idx)` / `get_layer_bias(idx)` — cloned tensors for a hidden layer.
- `get_all_weights()` — returns `List[Tensor]` for all layers in order (hidden layers first, output layer last). Shape per tensor: `(out_neurons, in_neurons)` — same as `nn.Linear.weight`.
- `forward_with_activations(x)` — single forward pass under `torch.no_grad()` that returns `(logits, activations)` where `activations` is a list: `[input_784, hidden_1_relu, ..., output_logits_10]`. Use this to power the animated architecture visualization after every prediction.
- `count_parameters()` / `architecture_summary()` — introspection helpers.

### SKILL: Weight & Bias Visualization

- Visualize `layer.weight.data` as a heatmap (matplotlib `imshow` or plotly `go.Heatmap`).
- For the first hidden layer: reshape each neuron's weights to 28×28 to show what pattern it detects.
- For deeper layers: show as a 2D matrix heatmap with a diverging colormap (`RdBu`, centered at 0).
- Visualize `layer.bias.data` as a bar chart.
- Add colorbar and axis labels on all weight heatmaps.
- Explanation text should appear near each visualization (pedagogical context).

### SKILL: Drawing Canvas for Digit Input

- Use `streamlit-drawable-canvas` (`pip install streamlit-drawable-canvas`).
- Canvas config: `stroke_width=20`, `stroke_color="#FFFFFF"`, `background_color="#1a1a1a"`, `width=320, height=320`.
- After drawing, extract `canvas_result.image_data` (RGBA numpy array).
- Preprocessing pipeline:
  1. Take channel 0 (R) from RGBA → grayscale 320×320
  2. Resize to 28×28 using `PIL.Image.LANCZOS`
  3. Normalize: `(pixel/255 - 0.1307) / 0.3081`
  4. Convert to float32 tensor shape `(1, 1, 28, 28)` — flattened in `forward`
- Run `model.eval()` before inference.
- Canvas key trick for reset: `key=f"canvas_{st.session_state['attempt']}"` — incrementing `attempt` causes Streamlit to recreate the widget blank.

### SKILL: Predict Flow (Tab 1)

Tab 1 uses a **two-phase state machine** gated by `show_result`:

**FASE 1** (`show_result == False`):
- Show canvas (320×320) + right column with instructions.
- "🔮 Predecir" button: disabled while canvas is empty (`img_data[:,:,:3].sum() == 0`).
- On button press: preprocess image → run inference → store activations/probs/pred/image in `session_state` → set `show_result=True` → `st.rerun()`.

**FASE 2** (`show_result == True`):
- Render `render_network_html()` via `components.html()` — the network builds itself sequentially.
- Below: predicted digit (big CSS number) + confidence bar + per-digit probability bars.
- "🔄 Intentar de nuevo" button: clears all result state, increments `attempt`, sets `show_result=False`, reruns → back to FASE 1 with blank canvas.

```python
# FASE 1 → FASE 2 transition
if predict_btn and has_drawing:
    ...  # inference
    st.session_state["show_result"] = True
    st.rerun()

# FASE 2 → FASE 1 transition (retry button)
if st.button("🔄 Intentar de nuevo"):
    st.session_state["show_result"] = False
    st.session_state["attempt"] += 1   # resets canvas key
    st.rerun()
```

### SKILL: Metrics Charts

All static chart functions live in `visualization/plots.py`:
- `plot_loss_curve(train_losses, val_losses)` → Plotly line chart
- `plot_accuracy_curve(train_accs, val_accs)` → Plotly line chart
- `plot_weight_heatmap(weight_tensor, layer_idx)` → matplotlib figure
- `plot_first_layer_receptive_fields(weight_tensor, n_neurons=16)` → grid of 28×28 heatmaps
- `plot_bias_bars(bias_tensor, layer_idx)` → Plotly bar chart
- `plot_prediction_probabilities(probs, predicted_class)` → Plotly bar chart (highlighted)
- `plot_network_architecture(...)` → Plotly static network graph (kept for reference; active visualization uses `network_html.py`)
- All functions return the chart object (figure) — let `app.py` call `st.plotly_chart` or `st.pyplot`.

### SKILL: Animated HTML Canvas Network (`network_html.py`)

`render_network_html()` in `visualization/network_html.py` generates a self-contained HTML string that embeds a 60fps Canvas 2D animation via `components.html()`.

**Signature:**
```python
def render_network_html(
    hidden_layers_config: List[int],
    weights_list: Optional[List[torch.Tensor]] = None,
    activations: Optional[List[np.ndarray]] = None,
    output_probs: Optional[np.ndarray] = None,
    max_neurons_display: int = 12,
    predicted_class: Optional[int] = None,
    input_image: Optional[np.ndarray] = None,   # (28,28) raw pixel array
    height: int = 530,
) -> str:
```

**How it works:**
- Python serializes all network data as JSON and embeds it in the HTML string via `__DATA__` placeholder replacement.
- The HTML contains a `<canvas>` element and a vanilla JS animation loop running at 60fps using `requestAnimationFrame`.
- On each Predict press, Streamlit rerenders `components.html()` with fresh data → JS always starts from frame 0 (automatic reset).

**Sequential Build Animation (phase-based state machine):**

| JS Constant | Value | Purpose |
|-------------|-------|---------|
| `T_BG` | 30 ticks | Background/grid/labels fade in |
| `T_LAYER` | 38 ticks | Node pop-in duration per layer |
| `T_EDGES` | 52 ticks | Edge draw duration per layer pair |

- `layerStart[li]` — tick at which layer `li` nodes start appearing (computed dynamically).
- `edgeStart[li]` — tick at which edges between layer `li` and `li+1` start drawing.
- `PULSE_START` — tick at which predicted output node begins 3× green pulse rings.
- `REVEAL_TOTAL` — tick at which full interactive mode begins.
- `revealTick` increments every frame. `REVEALED` flips to `true` when complete.

**JS helper functions:**
- `easeOut(t)` = `1 - (1-t)^3` — smooth deceleration for edge drawing.
- `easeOutBack(t)` — spring overshoot for node pop-in.
- `drawEdgesForPair(li, edgeProg, fullAlpha)` — draws edges at partial length (`edgeProg * full_length`).
- `drawNodesForLayer(li, nodeProg, burstProg)` — spring pop-in radius, burst ring on first appearance.
- `drawBurst(x, y, col, burstProg)` — expanding + fading ring.
- `drawPredictedPulse()` — 3 green expanding rings at predicted output node.
- `buildParticles()` — called only when `REVEALED=true`; particles flow along weighted edges.

**Visual features (post-reveal interactive mode):**
- Dark background (`#0f1117`) with subtle grid.
- Nodes glow with brightness proportional to activation magnitude.
- Edges color-coded: blue = positive weight, orange = negative weight; width ∝ magnitude.
- Animated particles flow left→right along edges, colored by weight sign.
- Hover tooltips on nodes: shows activation value, ReLU state, layer name.
- Predicted output node pulses green and stays highlighted.

**How to call in `app.py`:**
```python
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
```

### SKILL: Interactive Architecture Graph (Plotly, legacy)

`plot_network_architecture()` in `visualization/plots.py` renders a static Plotly network graph. Still in codebase for reference; the primary visualization is the HTML Canvas version.

Color palette constants (defined at module level in `plots.py`):
```python
_C_INPUT  = "#A8C4E0"   # light blue
_C_HIDDEN = "#9BA8B5"   # blue-gray
_C_OUTPUT = "#E88080"   # salmon
_C_PRED   = "#2ECC71"   # green (predicted)
_C_POS    = "#5C7BC4"   # positive weight
_C_NEG    = "#E8923A"   # negative weight
```

Helper `_hex_to_rgba(hex_color, alpha)` converts hex + float alpha to `"rgba(r,g,b,a)"` for per-node color encoding.

NaN-separator trick: all edges of the same weight bucket are a single `go.Scatter` trace, segments separated by `None`, for efficient rendering.

### SKILL: UI Layout

Streamlit layout structure in `app.py`:
```
Sidebar: hyperparameter sliders + Train button + status badge
Tabs: ["✏️ Dibujá y predecí", "📈 Curvas de entrenamiento", "🔬 Pesos del modelo"]
```
- **Sidebar**: n_hidden slider → per-layer neuron sliders (dynamic), lr select_slider, epochs slider, batch_size select_slider, Train button, status badge.
- **Tab 1 (Draw & Predict)**: Two-phase flow — FASE 1: canvas + Predict button; FASE 2: animated network + result + retry. The animated network visualization is embedded here (not a separate tab).
- **Tab 2 (Curves)**: 3 `st.metric` KPIs + loss curve + accuracy curve + expander with explanation.
- **Tab 3 (Weights)**: receptive field grid (1st layer) + per-layer weight heatmap + bias bar chart.

Use `st.columns` for side-by-side layouts.
Use `st.expander` for pedagogical explanations.
Use `st.metric` for displaying epoch, loss, accuracy KPIs.

### SKILL: Code Quality Rules

- All functions must have a single-line docstring.
- Use type hints on all function signatures.
- No magic numbers — define constants at top of file (e.g., `INPUT_SIZE = 784`, `OUTPUT_SIZE = 10`).
- Imports: stdlib first, then third-party, then local.
- Max line length: 100 chars.
- No `print()` in production code — use `st.write` or logging.
- Comment every non-obvious block with a `#` explaining *why*, not *what*.

### SKILL: Dependency Management

`requirements.txt` must pin these and only these:
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.32.0
matplotlib>=3.7.0
plotly>=5.18.0
Pillow>=10.0.0
numpy>=1.24.0
streamlit-drawable-canvas>=0.9.3
```

- Do NOT pin exact patch versions (`==`) — use `>=` for flexibility.
- Do NOT add unnecessary dependencies.

### SKILL: Error Handling & UX Guards

- Guard against prediction before training: show `st.info("👈 Entrenà el modelo desde el panel izquierdo para empezar.")` and `st.stop()`.
- Guard against drawing prediction on empty canvas: disable the Predict button when `img_data[:,:,:3].sum() == 0`.
- Guard against `streamlit-drawable-canvas` not installed: check `CANVAS_OK` flag and show `st.error`.
- Use `st.success` / `st.error` for completion/failure messages during training.

### SKILL: Performance Considerations

- MNIST download happens once; cache with `@st.cache_data` or store in session_state.
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` but default gracefully to CPU.
- Batch size slider: min=16, max=512, default=64.
- Training runs **synchronously** in Streamlit (no threads needed) using in-loop placeholder updates.
- For large epoch counts (>20), consider reducing visualization update frequency (every N epochs).
- The HTML Canvas animation is self-contained — no Python computation happens during animation. All data is serialized once to JSON at render time.

### SKILL: Pedagogical Annotations

Each visualization section in the UI must include an `st.expander` block explaining:
- What the metric/visualization represents.
- What to look for (e.g., "si el loss de validación sube mientras el de entrenamiento baja, hay overfitting").
- How hyperparameters affect it.

The network animation expander explains: node colors, edge colors, particles, glow = activation magnitude, predicted node pulse.

---

## Conventions Summary

| Concern | Choice | Reason |
|---|---|---|
| Model building | `nn.ModuleList` | Dynamic layer count |
| Loss | `CrossEntropyLoss` | Multiclass standard |
| Optimizer | `Adam` | Forgiving, adaptive |
| Normalization | mean=0.1307, std=0.3081 | MNIST standard |
| Static charts | Plotly (primary) + matplotlib (weight grids) | Plotly = interactive; matplotlib = pixel grids |
| Network animation | HTML Canvas 2D + vanilla JS (60fps) | Full control over animation, particles, glow effects |
| Canvas | `streamlit-drawable-canvas` | Best Streamlit-native drawing component |
| State | `st.session_state` only | No globals |
| Layout | 3 tabs | Clean separation; animation lives in Tab 1 |
| Predict UX | Explicit button + two-phase state machine | Deliberate, cinematic — user controls the moment |

---

## Anti-patterns to Avoid

- Do NOT use `st.experimental_rerun()` — deprecated; use `st.rerun()`.
- Do NOT call `model.train()` during inference — always `model.eval()`.
- Do NOT forget `optimizer.zero_grad()` before `loss.backward()`.
- Do NOT use `softmax` output with `CrossEntropyLoss` — it applies `log_softmax` internally.
- Do NOT hardcode layer sizes — always read from user config.
- Do NOT block the UI with a training loop that has no intermediate updates.
- Do NOT use `marker.opacity` for per-node brightness in Plotly — `opacity` is trace-level. Instead encode alpha inside each color string as `rgba(r,g,b,a)`.
- Do NOT create one `go.Scatter` trace per edge in the architecture graph — use the NaN-separator trick (single trace per bucket, segments separated by `None`) to keep trace count low and rendering fast.
- Do NOT use `forward_with_activations` inside the training loop — it is for inference only and runs under `torch.no_grad()`.
- Do NOT show raw logits as output node brightness — use `output_probs` (softmax) so values are bounded [0, 1] and visually meaningful.
- Do NOT set `update_streamlit=True` on the canvas if you want to avoid triggering a rerun on every stroke — currently it is `True` to enable the empty-canvas detection for disabling the Predict button.
- Do NOT call `buildParticles()` before `REVEALED=true` in the JS animation — particles must only start after the sequential build completes.
- Do NOT use a separate Architecture tab — the animated network lives inside Tab 1 (Draw & Predict) in FASE 2.
- Do NOT add right/wrong scoring — if the model predicts wrong, the user simply retrains. Keep the UX simple.

---

## Testing the App

Manual checklist before delivering:
1. Train model: sidebar → 2 hidden layers [128, 64], lr=0.001, epochs=5, batch=64 → Train button → live loss/accuracy updates appear.
2. Go to **Curvas** tab → loss and accuracy curves render with train/val lines.
3. Go to **Pesos** tab → receptive field grid (16 neurons) + weight heatmap + bias bars render.
4. Go to **Dibujá y predecí** tab → canvas is blank, "🔮 Predecir" button is disabled.
5. Draw a digit → Predict button becomes active.
6. Press Predict → FASE 2: sequential network build animation starts (labels fade in → input nodes pop → edges draw → hidden nodes pop → output lights up with green pulse on predicted digit).
7. After ~3–4s animation completes: particles flow, hover tooltips work on nodes.
8. Result section below: big predicted digit + confidence bar + per-digit probability bars.
9. Press "🔄 Intentar de nuevo" → canvas resets blank, animation disappears, back to FASE 1.
10. Test with 1 hidden layer and 4 hidden layers → animation phases adjust dynamically.

Regression checks:
- `get_all_weights()` must return `n_hidden + 1` tensors; last tensor shape `(10, last_hidden_size)`.
- `forward_with_activations(x)` must return `n_hidden + 2` activation arrays; first shape `(784,)`, last shape `(10,)`.
- `render_network_html(...)` must return a non-empty string with a `<canvas>` element.
- Incrementing `attempt` must produce a blank canvas (key change forces widget recreation).
