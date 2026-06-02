# CLAUDE.md — MNIST Neural Net Educational App

This file defines all skills, conventions, and guidance for Claude Code to assist on this project at the highest level.

---

## Project Purpose

Build a fully functional, educational, interactive Python application for training and visualizing a neural network that recognizes handwritten digits (MNIST) and letters (EMNIST A-Z). The goal is pedagogical: users learn *how* training works, not just get predictions.

---

## Repository

- **GitHub**: https://github.com/BrunoPignanelli/ML-competencia-
- **Branches**: `main` (production), `develop` (feature work), `testing` (QA)
- Always push all branches together after changes to `main`.

---

## Skills

### SKILL: Architecture Design

- The neural network class (`NeuralNet`) must be **dynamically configurable** at runtime: hidden layer count, neurons per layer, learning rate, batch size, epochs.
- Use `nn.ModuleList` to build layers dynamically — not hardcoded `nn.Sequential`.
- Use `CrossEntropyLoss` (multiclass standard — works for 10 classes digits or 26 classes letters).
- Use `Adam` optimizer (adaptive, forgiving for beginners).
- Normalize MNIST with `mean=0.1307, std=0.3081`. EMNIST Letters with `mean=0.1722, std=0.3309`.
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
│   └── neural_net.py             # NeuralNet class + compute_saliency()
├── training/
│   ├── __init__.py
│   └── trainer.py                # Training loop, metrics collection
├── data/
│   ├── __init__.py
│   └── loader.py                 # MNIST + EMNIST Letters DataLoaders
└── visualization/
    ├── __init__.py
    ├── plots.py                  # Plotly/matplotlib chart functions + saliency overlay
    └── network_html.py           # HTML/Canvas 60fps animated network component
```

- Never mix training logic into `app.py`.
- Never put visualization logic inside the model (exception: `compute_saliency` lives on `NeuralNet` because it requires access to the model's forward pass and gradient graph).
- `app.py` only calls functions from other modules and manages `st.session_state`.
- `plots.py` handles static Plotly/matplotlib charts (curves, heatmaps, bars, saliency overlay).
- `network_html.py` handles the animated 60fps Canvas network visualization.

### SKILL: Streamlit Session State Management

- Store all training results in `st.session_state` so they persist across reruns.
- **Mode key:**
  - `st.session_state["mode"]` — `"digits"` or `"letters"`, set by sidebar radio button

- **Digits keys (no suffix):**
  - `model`, `trainer`, `trained`, `test_acc`
  - `train_losses`, `val_losses`, `train_accs`, `val_accs`
  - `last_activations`, `last_probs`, `last_pred`, `last_input_image`, `last_saliency`
  - `show_result`, `attempt`
  - `confusion_matrix` — normalized (10×10) ndarray, computed after every train
  - `tsne_coords`, `tsne_labels` — 2D t-SNE coords (n,2) + class labels (n,); cleared on retrain

- **Letters keys (`_letters` suffix, mirrors digits exactly):**
  - `model_letters`, `trainer_letters`, `trained_letters`, `test_acc_letters`
  - `train_losses_letters`, `val_losses_letters`, `train_accs_letters`, `val_accs_letters`
  - `last_activations_letters`, `last_probs_letters`, `last_pred_letters`, `last_input_image_letters`, `last_saliency_letters`
  - `show_result_letters`, `attempt_letters`
  - `confusion_matrix_letters` — normalized (26×26) ndarray
  - `tsne_coords_letters`, `tsne_labels_letters` — t-SNE for letters mode

- **Suffix pattern** — all mode-aware code derives the suffix once and uses f-strings:
  ```python
  sfx = "_letters" if mode == "letters" else ""
  model = st.session_state[f"model{sfx}"]
  ```

- Never use global variables; always read/write via `st.session_state`.
- Use `st.rerun()` only after state mutations that require UI refresh.

### SKILL: Training Loop Pattern

The trainer must:
1. Accept a `NeuralNet` model, `DataLoader`, optimizer config, and a **callback** function.
2. Call the callback after each epoch with `(epoch, metrics_dict)`.
3. The callback updates `st.session_state` lists and updates placeholders — no `st.rerun()` inside the loop.
4. Return a history dict at the end.
5. Use `torch.no_grad()` for validation passes.
6. Compute accuracy as `correct / total * 100`.

**Retrain feature:** Second press of Train button with same architecture → button label changes to `"🔁 Seguir entrenando (30 épocas)"` → continues training the existing model (reuses Adam optimizer, momentum carries over). Detection:
```python
same_arch = (
    st.session_state[f"trained{sfx}"]
    and existing_net is not None
    and existing_net.hidden_layers_config == hidden_layers
)
RETRAIN_EPOCHS = 30
```

### SKILL: Dynamic Model Construction

```python
class NeuralNet(nn.Module):
    def __init__(self, hidden_layers, input_size=784, output_size=10, dropout_rate=0.0):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = input_size
        for size in hidden_layers:
            self.hidden.append(nn.Linear(prev, size))
            prev = size
        self.output_layer = nn.Linear(prev, output_size)
```

- Always flatten input in `forward` with `x.view(x.size(0), -1)`.
- Return raw logits (not softmax) — `CrossEntropyLoss` expects logits.
- For prediction with probabilities: apply `F.softmax(logits, dim=1)`.
- `output_size=10` for digits, `output_size=26` for letters — same class, different instantiation.

**Key methods on `NeuralNet`:**
- `predict_proba(x)` — softmax probabilities under `torch.no_grad()`.
- `get_layer_weights(idx)` / `get_layer_bias(idx)` — cloned tensors for a hidden layer.
- `get_all_weights()` — `List[Tensor]` for all layers (hidden + output), shape `(out, in)` each.
- `forward_with_activations(x)` — under `torch.no_grad()`, returns `(logits, [input_784, hidden_1_relu, ..., logits_n])`.
- `compute_saliency(x, target_class)` — vanilla gradient saliency, returns `(28,28)` ndarray normalized to [0,1].
- `extract_embeddings(loader, n_samples=1000)` — last hidden layer activations for n_samples; returns `(embeddings, labels)` ndarrays. Used for t-SNE. Runs under `torch.no_grad()`.
- `count_parameters()` / `architecture_summary()` — introspection helpers.
- `hidden_layers_config` — stored list of neuron counts, used for architecture comparison on retrain.

### SKILL: Data Loading (MNIST + EMNIST Letters)

Both loaders live in `data/loader.py` and follow identical structure.

**MNIST:**
```python
MNIST_MEAN = 0.1307 ; MNIST_STD = 0.3081 ; DATA_DIR = "./mnist_data"
load_mnist(batch_size=64, val_split=0.1, num_workers=0) → (train, val, test) DataLoaders
```

**EMNIST Letters:**
```python
EMNIST_MEAN = 0.1722 ; EMNIST_STD = 0.3309 ; EMNIST_DATA_DIR = "./emnist_data"
load_emnist_letters(batch_size=64, val_split=0.1, num_workers=0) → (train, val, test) DataLoaders
```

Critical EMNIST detail: labels are **1-indexed** (1=A … 26=Z). A `_ZeroIndexedWrapper` dataset subclass subtracts 1 from every label at `__getitem__` time, making them 0-indexed (0=A … 25=Z).

```python
class _ZeroIndexedWrapper(Dataset):
    def __getitem__(self, idx):
        img, label = self._ds[idx]
        return img, label - 1
```

### SKILL: Drawing Canvas for Digit/Letter Input

- Use `streamlit-drawable-canvas` (`pip install streamlit-drawable-canvas`).
- Canvas config: `stroke_width=20`, `stroke_color="#FFFFFF"`, `background_color="#1a1a1a"`, `width=320, height=320`.
- After drawing, extract `canvas_result.image_data` (RGBA numpy array).
- Preprocessing pipeline:
  1. Take channel 0 (R) from RGBA → grayscale 320×320
  2. Resize to 28×28 using `PIL.Image.LANCZOS`
  3. Normalize: `(pixel/255 - NORM_MEAN) / NORM_STD` — constants differ per mode
  4. Convert to float32 tensor shape `(1, 1, 28, 28)` — flattened in `forward`
- Run `model.eval()` before inference.
- Canvas key trick for reset: `key=f"canvas_{mode}_{st.session_state[f'attempt{sfx}']}"` — includes mode string to prevent cross-mode canvas bleed; incrementing `attempt` causes Streamlit to recreate widget blank.

### SKILL: Predict Flow (Tab 1 — Two-Phase State Machine)

Tab 1 is gated by `show_result{sfx}`. At the top of the tab, derive mode constants:

```python
_mode = st.session_state["mode"]
_sfx  = "_letters" if _mode == "letters" else ""
_norm_mean, _norm_std = (EMNIST_MEAN, EMNIST_STD) if _mode == "letters" else (0.1307, 0.3081)
_out_sz      = 26 if _mode == "letters" else 10
_class_labels = [chr(65+i) for i in range(26)] if _mode == "letters" else [str(i) for i in range(10)]
```

**FASE 1** (`show_result == False`):
- Canvas + "🔮 Predecir" button (disabled when canvas is empty).
- On press: preprocess → `predict_proba` → `forward_with_activations` → `compute_saliency` → store all in session state → `show_result=True` → `st.rerun()`.

**FASE 2** (`show_result == True`):
1. `render_network_html(...)` → `components.html()` — sequential build animation.
2. `st.expander("🔍 ¿En qué se fijó la red?")` → `plot_saliency_overlay(img_28, saliency, display_char)` → `st.pyplot()`.
3. Big predicted char (`chr(65+pred)` for letters, `str(pred)` for digits) + confidence bar.
4. Per-class probability bars, labeled with `_class_labels[i]`.
5. "🔄 Intentar de nuevo" — clears all `last_*{sfx}` keys, increments `attempt{sfx}`, reruns.

### SKILL: Saliency Map (Pixel Importance Visualization)

**What it is:** After a prediction, show which pixels drove the decision using vanilla gradient saliency. The user sees exactly what the network "looked at" in their drawing.

**Algorithm — implemented as `NeuralNet.compute_saliency()`:**
```python
def compute_saliency(self, x: torch.Tensor, target_class: int) -> np.ndarray:
    self.eval()
    inp = x.clone().detach().requires_grad_(True)  # clone — never mutate original
    output = self.forward(inp)
    self.zero_grad()
    output[0, target_class].backward()             # gradient of target logit w.r.t. input
    saliency = inp.grad.data.abs().squeeze()       # abs: sign irrelevant, magnitude = importance
    sal_np = saliency.cpu().numpy()
    return (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)  # normalize [0,1]
```

**Visualization — `plot_saliency_overlay()` in `visualization/plots.py`:**
- 3-panel matplotlib figure on dark (`#0f1117`) background.
- Panel 1: original drawing (`cmap="gray"`).
- Panel 2: saliency heatmap (`cmap="hot"` — black→red→yellow→white = low→high importance).
- Panel 3: overlay (drawing at `alpha=0.35` + saliency at `alpha=0.75`).
- Title of panel 3 is green `#2ECC71` and shows the predicted class label.
- Call `plt.close(fig)` after `st.pyplot(fig)` to free memory.

**Anti-patterns for saliency:**
- Do NOT call inside `torch.no_grad()` — gradients won't flow.
- Do NOT modify the original `tensor` — always `.clone().detach()` first.
- Do NOT normalize before taking abs — take abs, then normalize.
- Do NOT use `"jet"` colormap — it misleads about magnitude ordering. Use `"hot"` or `"plasma"`.

### SKILL: Metrics Charts

All static chart functions live in `visualization/plots.py` and return the figure object (never render):
- `plot_loss_curve(train_losses, val_losses)` → Plotly line chart
- `plot_accuracy_curve(train_accs, val_accs)` → Plotly line chart
- `plot_weight_heatmap(weight_tensor, layer_name)` → matplotlib figure
- `plot_first_layer_receptive_fields(weight_tensor, n_neurons=16)` → grid of 28×28 heatmaps
- `plot_bias_bars(bias_tensor, layer_name)` → Plotly bar chart
- `plot_prediction_probabilities(probs, predicted_class)` → Plotly bar chart (highlighted)
- `plot_saliency_overlay(input_image, saliency, class_label)` → matplotlib 3-panel figure
- `plot_confusion_matrix(cm, class_labels)` → Plotly heatmap, normalized by row (recall per class); diagonal highlighted with white square markers; height scales with n_classes.
- `plot_tsne(coords, labels, class_labels)` → Plotly scatter, one trace per class, dark background; 26-color palette supports both digit and letter modes.
- `plot_network_architecture(...)` → Plotly static graph (legacy reference; active viz is HTML Canvas)

### SKILL: Animated HTML Canvas Network (`network_html.py`)

`render_network_html()` generates a self-contained HTML string with a 60fps Canvas 2D animation.

**Full signature:**
```python
def render_network_html(
    hidden_layers_config: List[int],
    weights_list: Optional[List[torch.Tensor]] = None,
    activations: Optional[List[np.ndarray]] = None,
    output_probs: Optional[np.ndarray] = None,
    max_neurons_display: int = 12,
    predicted_class: Optional[int] = None,
    input_image: Optional[np.ndarray] = None,
    height: int = 530,
    output_size: int = 10,           # 10 for digits, 26 for letters
    class_labels: Optional[List[str]] = None,  # ["0"…"9"] or ["A"…"Z"]
) -> str:
```

Data is serialized as JSON and injected via `__DATA__` placeholder. The JS uses `DATA.classLabels[ni]` for output node tooltips and labels — never hardcoded digit strings.

**Sequential Build Animation (phase-based state machine):**

| JS Constant | Value | Purpose |
|-------------|-------|---------|
| `T_BG` | 30 ticks | Background/grid/labels fade in |
| `T_LAYER` | 38 ticks | Node pop-in duration per layer |
| `T_EDGES` | 52 ticks | Edge draw duration per layer pair |

- `layerStart[li]` / `edgeStart[li]` — computed dynamically per architecture.
- `revealTick` increments every frame; `REVEALED` flips to `true` at `REVEAL_TOTAL`.
- `buildParticles()` only called after `REVEALED=true`.
- Predicted output node: 3× green pulse rings (`drawPredictedPulse()`).

**Post-reveal interactive mode:**
- Dark background (`#0f1117`), nodes glow proportional to activation.
- Edges: blue = positive weight, orange = negative weight; width ∝ magnitude.
- Particles flow left→right along edges, colored by weight sign.
- Hover tooltips: activation value, ReLU state, layer name, class label.

### SKILL: Sidebar Live Architecture Preview

`_sidebar_preview_svg(hidden_layers, output_size=10)` in `app.py` renders a compact SVG network diagram that re-renders instantly as any slider changes.

```python
def _sidebar_preview_svg(hidden_layers: list, output_size: int = 10) -> str:
    all_layers = [784] + hidden_layers + [output_size]
    ...  # SVG circles per layer, edges, neuron count labels
```

Displayed via `st.markdown(svg_str, unsafe_allow_html=True)` + a `st.caption` showing total layers, hidden neurons, and parameter count. Passes `output_size=26` when mode is letters.

### SKILL: Mode Selector (Digits vs Letters)

At the top of the sidebar, before hyperparameters:

```python
st.radio("Módulo", options=["digits", "letters"],
    format_func=lambda m: "🔢 Dígitos (MNIST)" if m == "digits" else "🔡 Letras (EMNIST)",
    horizontal=True, key="mode")
mode = st.session_state["mode"]
```

All downstream code derives `sfx`, `out_sz`, `_norm_mean/_norm_std`, `_class_labels`, and `_draw_prompt` from `mode`. Modes are fully isolated — training one never affects the other's state.

### SKILL: UI Layout

```
Sidebar: mode radio → hyperparameter sliders → SVG preview → Train button → status badge → tips
Tabs: ["✏️ Dibujá y predecí", "📈 Curvas de entrenamiento", "🔬 Pesos del modelo"]
```

- **Tab 1**: FASE 1 (canvas + Predict) → FASE 2 (network animation → saliency map → result → retry).
- **Tab 2**: 3 `st.metric` KPIs + loss curve + accuracy curve + confusion matrix + t-SNE section.
- **Tab 3**: receptive field grid + per-layer weight heatmap + bias bars.

All tabs read state via the `sfx` suffix pattern so they show the correct module's data.

### SKILL: Code Quality Rules

- All functions must have a single-line docstring.
- Use type hints on all function signatures.
- No magic numbers — define constants at top of file.
- Imports: stdlib first, then third-party, then local.
- Max line length: 100 chars.
- No `print()` in production code.
- Comment every non-obvious block with `#` explaining *why*, not *what*.

### SKILL: Live Training Curves

During training, the `on_epoch` callback also updates a `ph_live_chart = st.empty()` placeholder in the sidebar with `plot_loss_curve(...)` starting from epoch 2 (needs ≥2 points for a line). This lets the user watch the loss fall in real time without leaving the sidebar. After training ends and `st.rerun()` fires, the full curves render in tab_curves as usual.

```python
ph_live_chart = st.empty()

def on_epoch(epoch, metrics):
    # ... metric updates ...
    if epoch >= 2:
        ph_live_chart.plotly_chart(
            plot_loss_curve(train_losses_list, val_losses_list),
            use_container_width=True,
        )
```

### SKILL: Confusion Matrix (Tab 2)

Computed automatically after every training run via `trainer.confusion_matrix(test_loader, n_classes=out_sz)` and stored in `st.session_state[f"confusion_matrix{sfx}"]`. Displayed in tab_curves below the loss/accuracy charts with `plot_confusion_matrix(cm, class_labels)`.

- Row = true class, column = predicted class, values = recall per class (row-normalized).
- 10×10 for digits, 26×26 for letters.
- Cleared and recomputed whenever the model is retrained.

### SKILL: t-SNE Visualization (Tab 2)

On-demand visualization of the last hidden layer's learned representations.

**Flow:**
1. User sets sample count with slider (200–2000, default 1000).
2. Clicks "🔭 Generar t-SNE" button.
3. App recreates the test loader, calls `net.extract_embeddings(loader, n_samples)`.
4. Runs `sklearn.manifold.TSNE(n_components=2, random_state=42, perplexity=min(30, n//10))`.
5. Stores `tsne_coords{sfx}` and `tsne_labels{sfx}` in session state and renders `plot_tsne(...)`.

Result is cached until the next retrain. Costs ~10–20s for 1000 samples.

**Anti-patterns:**
- Do NOT run t-SNE automatically after training — it's slow and the user may not need it.
- Do NOT store the DataLoader in session state — recreate it on demand.
- Do NOT use perplexity > n_samples/10 — sklearn will raise an error.

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
scipy>=1.10.0
scikit-learn>=1.0.0
```

- Do NOT pin exact patch versions (`==`) — use `>=`.
- Do NOT add unnecessary dependencies.

### SKILL: Error Handling & UX Guards

- Guard against prediction before training: `st.info(...)` + `st.stop()`.
- Guard against empty canvas: disable Predict button when `img_data[:,:,:3].sum() == 0`.
- Guard against missing canvas library: check `CANVAS_OK` flag.
- Clear all `last_*` state when retraining (prevents stale activations from old model crashing the HTML component with an IndexError when `layer_idx` exceeds activation array size).

### SKILL: Performance Considerations

- MNIST/EMNIST download cached on disk after first run.
- EMNIST first download is ~50 MB — show a distinct spinner message.
- Training runs synchronously using in-loop placeholder updates (no threads needed).
- Saliency: one backward pass per prediction — negligible cost.
- HTML Canvas animation: all data serialized once to JSON; zero Python cost during animation.
- Call `plt.close(fig)` after every `st.pyplot()` to prevent matplotlib figure accumulation.

### SKILL: Pedagogical Annotations

Each visualization section must include an `st.expander` explaining:
- What the metric/visualization represents.
- What to look for (overfitting signals, convergence, weight patterns).
- How hyperparameters affect it.

Saliency expander caption: `"Los píxeles blancos/amarillos influyeron más en la predicción. Los negros/rojos oscuros fueron ignorados."`

---

## Conventions Summary

| Concern | Choice | Reason |
|---|---|---|
| Model building | `nn.ModuleList` | Dynamic layer count |
| Loss | `CrossEntropyLoss` | Works for any class count (10 or 26) |
| Optimizer | `Adam` | Forgiving, adaptive; state saved for retrain |
| Normalization | Per-dataset constants | MNIST ≠ EMNIST stats |
| Static charts | Plotly (primary) + matplotlib (grids, saliency) | Plotly = interactive; matplotlib = pixel-level |
| Network animation | HTML Canvas 2D + vanilla JS (60fps) | Full control: particles, glow, sequential build |
| Canvas | `streamlit-drawable-canvas` | Best Streamlit-native drawing component |
| State | `st.session_state` only, suffix pattern | No globals; full mode isolation |
| Layout | 3 tabs + sidebar mode radio | Clean separation |
| Predict UX | Explicit button + two-phase state machine | Deliberate, cinematic |
| Explainability | Vanilla Gradient Saliency | No extra deps; works for any class count |
| Retrain UX | Same architecture → continue 30 epochs | Adam momentum preserved; faster convergence |
| Live curves | `st.empty()` placeholder updated in `on_epoch` | User watches loss fall without leaving sidebar |
| Confusion matrix | Computed post-train, stored in session state | Always fresh; cleared on retrain |
| t-SNE | On-demand via button; `sklearn.TSNE` | Avoids blocking training; result cached |

---

## Anti-patterns to Avoid

- Do NOT use `st.experimental_rerun()` — deprecated; use `st.rerun()`.
- Do NOT call `model.train()` during inference — always `model.eval()`.
- Do NOT forget `optimizer.zero_grad()` before `loss.backward()`.
- Do NOT use `softmax` output with `CrossEntropyLoss` — it applies `log_softmax` internally.
- Do NOT hardcode layer sizes or class counts — always read from config.
- Do NOT block the UI with a training loop that has no intermediate updates.
- Do NOT use `marker.opacity` for per-node brightness in Plotly — encode alpha inside `rgba()` color strings.
- Do NOT create one `go.Scatter` trace per edge — use the NaN-separator trick (single trace per bucket, `None`-separated segments).
- Do NOT use `forward_with_activations` inside the training loop — inference only, runs under `torch.no_grad()`.
- Do NOT show raw logits as output node brightness — use softmax `output_probs`.
- Do NOT call `buildParticles()` before `REVEALED=true` in the JS animation.
- Do NOT use a separate Architecture tab — the animated network lives in Tab 1 FASE 2.
- Do NOT add right/wrong scoring — user retrains if wrong.
- Do NOT call `compute_saliency` inside `torch.no_grad()` — gradients won't flow.
- Do NOT modify the original input tensor for saliency — always `.clone().detach()` first.
- Do NOT use `"jet"` colormap for saliency — it misleads magnitude ordering. Use `"hot"` or `"plasma"`.
- Do NOT forget `plt.close(fig)` after `st.pyplot(fig)` — matplotlib accumulates figures in memory.
- Do NOT skip clearing `last_saliency{sfx}` on retry/retrain — stale saliency from a different drawing would show.
- Do NOT use MNIST normalization constants for EMNIST inference — `0.1722/0.3309` for letters, `0.1307/0.3081` for digits.
- Do NOT run t-SNE automatically after training — always on-demand via button (too slow to block the UI).
- Do NOT store a DataLoader in session state — recreate it when needed (not serializable).
- Do NOT pass `perplexity > n_samples/10` to sklearn TSNE — it will raise a ValueError.
- Do NOT forget to clear `tsne_coords{sfx}` and `tsne_labels{sfx}` on retrain — stale embeddings from the old model are meaningless.

---

## Testing the App

Manual checklist before delivering:

**Digits flow:**
1. Sidebar → mode = 🔢 Dígitos → train [128, 64], lr=0.001, epochs=5, batch=64 → live progress shows.
2. Second train press with same arch → button says "🔁 Seguir entrenando (30 épocas)" → continues from existing weights.
3. Curvas tab → loss + accuracy curves render with full history (including retrain epochs appended).
4. Pesos tab → receptive fields + weight heatmap + bias bars render.
5. Dibujá tab → canvas blank, Predict button disabled.
6. Draw digit → Predict → FASE 2: network builds layer by layer → saliency map expander shows 3 panels → result shows big digit + confidence + probability bars.
7. Retry → blank canvas, FASE 1.

**Letters flow:**
8. Sidebar → mode = 🔡 Letras → train → EMNIST downloads (~50 MB first time).
9. Draw a letter → Predict → animation has 26 output nodes → result shows predicted letter A-Z.
10. Probability bars labeled A-Z.
11. Saliency map works identically.
12. Switch back to Digits → digits model + history still intact.

**Regression checks:**
- `get_all_weights()` returns `n_hidden + 1` tensors; last shape `(output_size, last_hidden_size)`.
- `forward_with_activations(x)` returns `n_hidden + 2` arrays; first shape `(784,)`, last shape `(output_size,)`.
- `compute_saliency(x, pred)` returns ndarray shape `(28,28)`, values in `[0,1]`.
- `extract_embeddings(loader, 500)` returns `(ndarray (500, last_hidden_size), ndarray (500,))`.
- `render_network_html(..., output_size=26, class_labels=["A",...,"Z"])` returns HTML with `<canvas>`.
- Incrementing `attempt_letters` resets letters canvas without affecting digits canvas.
- Retraining (any mode) clears `last_saliency{sfx}`, `show_result{sfx}`, `tsne_coords{sfx}`, `tsne_labels{sfx}` to prevent stale state crash.
- After training, `confusion_matrix{sfx}` is a float ndarray shape `(n_classes, n_classes)` with row sums ≈ 1.0.
- t-SNE button → `plot_tsne(coords, labels, class_labels)` renders Plotly scatter with one trace per class.
