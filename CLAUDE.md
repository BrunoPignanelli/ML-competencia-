# CLAUDE.md — MNIST Neural Net Educational App

This file defines all skills, conventions, and guidance for Claude Code to assist on this project at the highest level.

---

## Project Purpose

Build a fully functional, educational, interactive Python application for training and visualizing a neural network that recognizes handwritten digits (MNIST). The goal is pedagogical: users learn *how* training works, not just get predictions.

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
├── requirements.txt
├── app.py                  # Streamlit entry point
├── model/
│   ├── __init__.py
│   └── neural_net.py       # NeuralNet class (configurable)
├── training/
│   ├── __init__.py
│   └── trainer.py          # Training loop, metrics collection
├── data/
│   ├── __init__.py
│   └── loader.py           # MNIST download + DataLoader
└── visualization/
    ├── __init__.py
    └── plots.py            # All matplotlib/plotly chart functions
```

- Never mix training logic into `app.py`.
- Never put visualization logic inside the model.
- `app.py` only calls functions from other modules and manages `st.session_state`.

### SKILL: Streamlit Session State Management

- Store all training results in `st.session_state` so they persist across reruns.
- Key state keys:
  - `st.session_state["model"]` — trained NeuralNet instance
  - `st.session_state["train_losses"]` — list of per-epoch loss
  - `st.session_state["train_accs"]` — list of per-epoch accuracy
  - `st.session_state["val_losses"]` — list of per-epoch val loss
  - `st.session_state["val_accs"]` — list of per-epoch val accuracy
  - `st.session_state["trained"]` — bool flag
  - `st.session_state["last_activations"]` — `List[np.ndarray]` from last prediction (for arch viz)
  - `st.session_state["last_probs"]` — `np.ndarray` shape (10,) from last prediction
  - `st.session_state["last_pred"]` — `int` predicted class from last prediction
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
- `forward_with_activations(x)` — single forward pass under `torch.no_grad()` that returns `(logits, activations)` where `activations` is a list: `[input_784, hidden_1_relu, ..., output_logits_10]`. Use this to power the architecture graph after every prediction.
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
- Canvas config: `stroke_width=20`, `stroke_color="white"`, `background_color="black"`, `width=280, height=280`.
- After drawing, extract `canvas_result.image_data` (RGBA numpy array).
- Preprocessing pipeline:
  1. Take channel 0 (R) from RGBA → grayscale 280×280
  2. Resize to 28×28 using `PIL.Image.LANCZOS`
  3. Normalize: `(pixel/255 - 0.1307) / 0.3081`
  4. Convert to float32 tensor, add batch dim: `shape = (1, 1, 28, 28)` — but since model expects flat, it will be flattened in forward
  5. Alternatively: shape `(1, 784)` directly
- Run `model.eval()` before inference.

### SKILL: Metrics Charts

All chart functions live in `visualization/plots.py`:
- `plot_loss_curve(train_losses, val_losses)` → Plotly line chart
- `plot_accuracy_curve(train_accs, val_accs)` → Plotly line chart
- `plot_weight_heatmap(weight_tensor, layer_idx)` → matplotlib figure
- `plot_first_layer_receptive_fields(weight_tensor, n_neurons=16)` → grid of 28×28 heatmaps
- `plot_bias_bars(bias_tensor, layer_idx)` → plotly bar chart
- `plot_prediction_probabilities(probs, predicted_class)` → plotly bar chart (highlighted)
- `plot_network_architecture(...)` → Plotly interactive network graph (see SKILL below)
- All functions return the chart object (figure), not render it — let `app.py` call `st.plotly_chart` or `st.pyplot`.

### SKILL: Interactive Architecture Graph

`plot_network_architecture()` in `visualization/plots.py` renders a fully dynamic Plotly network graph.

Signature:
```python
def plot_network_architecture(
    hidden_layers_config: List[int],
    weights_list: Optional[List[torch.Tensor]] = None,   # from net.get_all_weights()
    activations: Optional[List[np.ndarray]] = None,      # from net.forward_with_activations()
    output_probs: Optional[np.ndarray] = None,           # shape (10,), softmax probs
    max_neurons_display: int = 12,
    predicted_class: Optional[int] = None,
) -> go.Figure:
```

Key design decisions:
- **Input layer**: always displays 10 representative nodes (sampled evenly from 784) plus a white rectangle representing the raw pixel grid. Lines fan out from the box to the 10 nodes.
- **Hidden layers**: shows `min(n_neurons, max_neurons_display)` nodes; shows `···` annotation when truncated.
- **Output layer**: always shows all 10 nodes. When `output_probs` is set, digit label + percentage appears to the right of each node.
- **Edges**: grouped into 4 opacity/width buckets by normalized weight magnitude (±0.4 threshold). Uses the **NaN-separator trick** — all edges of the same bucket are a single `go.Scatter` trace for efficiency, not one trace per edge.
  - Strong positive (|w| ≥ 0.4): blue, opacity 0.85, width 2.0
  - Weak positive (0 ≤ |w| < 0.4): blue, opacity 0.30, width 1.0
  - Weak negative: orange, opacity 0.30, width 1.0
  - Strong negative (|w| ≥ 0.4): orange, opacity 0.85, width 2.0
  - No weights available: neutral gray, opacity 0.18, width 0.8
- **Node brightness**: opacity encoded as `rgba()` color strings (per-node, not per-trace). `_activation_opacities()` normalizes activations to [0.20, 1.0]. For output nodes, uses `output_probs` instead of raw logits.
- **Predicted class**: output node color changes to `#2ECC71` (green).
- **Legend**: uses `x=[None], y=[None]` invisible data traces so Plotly's built-in horizontal legend box renders correctly without polluting the plot area.
- **Pre-training preview**: when `weights_list=None`, neutral gray connections are drawn from the sidebar's current `hidden_layers` config — users see the structure before training.

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

How to wire it up in `app.py`:
```python
# In prediction block, after probs:
_, acts = net.forward_with_activations(tensor)
st.session_state["last_activations"] = acts
st.session_state["last_probs"] = probs
st.session_state["last_pred"] = pred

# In architecture tab:
fig = plot_network_architecture(
    hidden_layers_config=net.hidden_layers_config,
    weights_list=net.get_all_weights(),
    activations=st.session_state["last_activations"],
    output_probs=st.session_state["last_probs"],
    predicted_class=st.session_state["last_pred"],
)
st.plotly_chart(fig, use_container_width=True)
```

### SKILL: UI Layout

Streamlit layout structure in `app.py`:
```
Sidebar: hyperparameter sliders + Train button + status badge
Tabs: ["✏️ Dibujá y predecí", "📈 Curvas de entrenamiento", "🔬 Pesos del modelo", "🧠 Arquitectura"]
```
- **Sidebar**: n_hidden slider → per-layer neuron sliders (dynamic), lr select_slider, epochs slider, batch_size select_slider, Train button, status badge.
- **Tab 1 (Draw & Predict)**: `st_canvas` left, prediction output right. Prediction wires activations into `session_state` for the architecture tab.
- **Tab 2 (Curves)**: 3 `st.metric` KPIs + loss curve + accuracy curve + expander with explanation.
- **Tab 3 (Weights)**: receptive field grid (1st layer) + per-layer weight heatmap + bias bar chart.
- **Tab 4 (Architecture)**: `plot_network_architecture` graph + `max_neurons_display` slider + expander with reading guide.

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

- Guard against prediction before training: show `st.warning("Primero entrená el modelo.")`.
- Guard against drawing prediction on empty canvas: check if `canvas_result.image_data` is not all zeros.
- Guard against MNIST not loaded: show `st.info("Cargá el dataset primero.")`.
- Use `st.spinner("Descargando MNIST...")` for the data loading step.
- Use `st.success` / `st.error` for completion/failure messages.

### SKILL: Performance Considerations

- MNIST download happens once; cache with `@st.cache_data` or store in session_state.
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` but default gracefully to CPU.
- Batch size slider: min=16, max=512, default=64.
- Training runs **synchronously** in Streamlit (no threads needed) using in-loop placeholder updates.
- For large epoch counts (>20), consider reducing visualization update frequency (every N epochs).

### SKILL: Pedagogical Annotations

Each visualization section in the UI must include an `st.expander("¿Qué muestra esta gráfica?")` block explaining:
- What the metric/visualization represents
- What to look for (e.g., "si el loss de validación sube mientras el de entrenamiento baja, hay overfitting")
- How hyperparameters affect it

---

## Conventions Summary

| Concern | Choice | Reason |
|---|---|---|
| Model building | `nn.ModuleList` | Dynamic layer count |
| Loss | `CrossEntropyLoss` | Multiclass standard |
| Optimizer | `Adam` | Forgiving, adaptive |
| Normalization | mean=0.1307, std=0.3081 | MNIST standard |
| Charts | Plotly (primary) + matplotlib (weight grids) | Plotly = interactive; matplotlib = pixel grids |
| Canvas | `streamlit-drawable-canvas` | Best Streamlit-native drawing component |
| State | `st.session_state` only | No globals |
| Layout | Tabs | Clean separation of concerns |

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

---

## Testing the App

Manual checklist before delivering:
1. Load MNIST → no errors
2. Configure 2 hidden layers [128, 64], lr=0.001, epochs=5, batch=64
3. Train → see live loss drop, accuracy rise
4. Go to **Curvas** tab → loss and accuracy curves render
5. Go to **Pesos** tab → receptive field grid + weight heatmap + bias bars render
6. Go to **Predicción** tab → draw a digit → prediction shows class + probability bars
7. Go to **Arquitectura** tab → graph shows weight-colored connections (blue/orange)
8. Draw another digit → go back to Arquitectura tab → node brightness updates, predicted output node turns green, probability labels appear on the right
9. Change to 3 hidden layers, retrain → architecture graph updates to 3 hidden columns
10. Before training: architecture tab shows gray neutral connections as a structural preview

Architecture graph regression checks:
- `plot_network_architecture(hidden_layers_config=[128])` must render without weights (neutral gray)
- `get_all_weights()` must return `n_hidden + 1` tensors; last tensor shape `(10, last_hidden_size)`
- `forward_with_activations(x)` must return `n_hidden + 2` activation arrays; first shape `(784,)`, last shape `(10,)`
