"""
landing_html.py — Página de bienvenida animada para Neural Decode.

HTML autocontenido con 4 secciones scrolleables:
  1. Hero  — canvas de red neuronal + título
  2. Features — 3 cards glassmorphism
  3. How It Works — 4 pasos con conectores animados
  4. CTA  — botón de entrada

IMPORTANTE: el HTML usa JS para leer window.innerHeight en lugar de vh,
porque dentro de un iframe de Streamlit los vh se calculan sobre la altura
total del iframe (ej. 4000px), no sobre el viewport visible.
"""


def render_landing_html() -> str:
    """Retorna el HTML completo de la landing page como string."""
    return r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Neural Decode</title>
<style>
/* ── Reset ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: #04050f;
  color: #fff;
  overflow-x: hidden;
  scroll-behavior: smooth;
}

/* ── Sections — height set by JS ────────────────────────────────────── */
.full-section {
  width: 100%;
  /* min-height is set dynamically via JS to window.innerHeight */
  min-height: 600px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

/* ── Scroll reveal ──────────────────────────────────────────────────── */
.reveal {
  opacity: 0;
  transform: translateY(36px);
  transition: opacity 0.8s cubic-bezier(.22,1,.36,1),
              transform 0.8s cubic-bezier(.22,1,.36,1);
}
.reveal.visible { opacity: 1; transform: translateY(0); }
.reveal.d1 { transition-delay: 0.10s; }
.reveal.d2 { transition-delay: 0.22s; }
.reveal.d3 { transition-delay: 0.36s; }
.reveal.d4 { transition-delay: 0.50s; }

/* ── Gradient text ──────────────────────────────────────────────────── */
.grad {
  background: linear-gradient(135deg, #36D1DC 0%, #5B86E5 50%, #2ECC71 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-size: 200% 200%;
  animation: gradShift 5s ease-in-out infinite alternate;
}
@keyframes gradShift {
  0%   { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

/* ════════════════════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════════════════════ */
#hero {
  background: radial-gradient(ellipse at 55% 35%, #0a1535 0%, #04050f 65%);
  padding: 0;
}
#heroCanvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
}
/* Dark overlay so text is always readable over the canvas */
#heroOverlay {
  position: absolute;
  inset: 0;
  z-index: 2;
  background:
    radial-gradient(ellipse at 50% 50%, rgba(4,5,15,0.05) 0%, rgba(4,5,15,0.7) 100%);
  pointer-events: none;
}
.hero-content {
  position: relative;
  z-index: 3;
  text-align: center;
  padding: 0 32px;
  max-width: 860px;
  width: 100%;
}
.hero-eyebrow {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 5px;
  color: #5B86E5;
  text-transform: uppercase;
  margin-bottom: 24px;
  opacity: 0;
  animation: fadeUp 0.8s 0.3s ease forwards;
}
.hero-title {
  font-size: clamp(48px, 7vw, 88px);
  font-weight: 900;
  line-height: 1.0;
  letter-spacing: -2px;
  margin-bottom: 10px;
  opacity: 0;
  animation: fadeUp 0.9s 0.5s ease forwards;
}
.hero-rule {
  display: block;
  height: 3px;
  width: 0;
  margin: 0 auto 28px;
  border-radius: 2px;
  background: linear-gradient(90deg, #2ECC71, #5B86E5);
  animation: growRule 1s 1.2s ease forwards;
}
.hero-sub {
  font-size: clamp(15px, 2vw, 19px);
  color: #9ba8c9;
  line-height: 1.7;
  max-width: 520px;
  margin: 0 auto 36px;
  opacity: 0;
  animation: fadeUp 0.9s 0.72s ease forwards;
}
.hero-pills {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
  opacity: 0;
  animation: fadeUp 0.9s 0.9s ease forwards;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 100px;
  padding: 8px 18px;
  font-size: 13px;
  color: #ccd6f6;
  backdrop-filter: blur(8px);
}
.pill-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #2ECC71;
  box-shadow: 0 0 8px rgba(46,204,113,0.7);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.6; transform:scale(0.75); }
}
.scroll-indicator {
  position: absolute;
  bottom: 32px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  opacity: 0;
  animation: fadeUp 1s 1.5s ease forwards;
}
.scroll-label {
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #404870;
}
.scroll-arrow {
  animation: arrowBounce 1.8s 2s ease-in-out infinite;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes growRule {
  from { width:0; }
  to   { width:80px; }
}
@keyframes arrowBounce {
  0%,100% { transform:translateY(0); }
  50%     { transform:translateY(8px); }
}

/* ════════════════════════════════════════════════════════════════════
   FEATURES
════════════════════════════════════════════════════════════════════ */
#features {
  background: linear-gradient(180deg, #04050f 0%, #060917 100%);
  padding: 80px 40px;
}
#features::before {
  content: '';
  position: absolute;
  top: -80px; left: 50%; transform: translateX(-50%);
  width: 800px; height: 400px;
  background: radial-gradient(ellipse, rgba(91,134,229,0.09) 0%, transparent 65%);
  pointer-events: none;
}
.section-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 5px;
  text-transform: uppercase;
  color: #5B86E5;
  margin-bottom: 12px;
  text-align: center;
}
.section-title {
  font-size: clamp(26px, 4vw, 42px);
  font-weight: 800;
  text-align: center;
  margin-bottom: 64px;
  letter-spacing: -0.8px;
}
.cards {
  display: flex;
  gap: 22px;
  flex-wrap: wrap;
  justify-content: center;
  max-width: 1080px;
  width: 100%;
}
.card {
  flex: 1;
  min-width: 270px;
  max-width: 340px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-top: 2px solid #2ECC71;
  border-radius: 18px;
  padding: 38px 30px 32px;
  backdrop-filter: blur(14px);
  transition: transform 0.3s ease, box-shadow 0.3s ease, border-top-color 0.3s;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(46,204,113,0.5), transparent);
}
.card:hover {
  transform: translateY(-7px);
  box-shadow: 0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px rgba(46,204,113,0.12);
  border-top-color: #36D1DC;
}
.card-icon {
  width: 44px; height: 44px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(91,134,229,0.08);
  border: 1px solid rgba(91,134,229,0.15);
}
.card-icon svg { width: 22px; height: 22px; }
.card-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 12px;
  color: #e6edf3;
}
.card-body { font-size: 14px; color: #8892b0; line-height: 1.75; }
.card-body strong { color: #a0aec0; }

/* ════════════════════════════════════════════════════════════════════
   HOW IT WORKS
════════════════════════════════════════════════════════════════════ */
#how {
  background: linear-gradient(180deg, #060917 0%, #050714 100%);
  padding: 80px 40px;
}
#how::after {
  content: '';
  position: absolute;
  bottom: -60px; right: -80px;
  width: 480px; height: 480px;
  background: radial-gradient(ellipse, rgba(46,204,113,0.06) 0%, transparent 65%);
  pointer-events: none;
}
.steps-row {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  max-width: 980px;
  width: 100%;
  margin-bottom: 72px;
  flex-wrap: wrap;
  gap: 0;
}
.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  flex: 1;
  min-width: 180px;
  max-width: 230px;
  padding: 0 12px;
}
.step-icon-wrap {
  width: 60px; height: 60px;
  border-radius: 50%;
  background: rgba(91,134,229,0.08);
  border: 1.5px solid rgba(91,134,229,0.25);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 20px;
  transition: background 0.5s, border-color 0.5s, box-shadow 0.5s;
}
.step-icon-wrap svg { width: 24px; height: 24px; }
.step.visible .step-icon-wrap {
  background: rgba(46,204,113,0.1);
  border-color: rgba(46,204,113,0.4);
  box-shadow: 0 0 24px rgba(46,204,113,0.15);
}
.step-heading { font-size: 15px; font-weight: 700; color: #ccd6f6; margin-bottom: 9px; }
.step-desc { font-size: 13px; color: #8892b0; line-height: 1.65; }
.connector {
  flex: 0 0 40px;
  height: 2px;
  margin-top: 29px;
  background: rgba(91,134,229,0.15);
  position: relative;
  overflow: hidden;
  border-radius: 1px;
}
.connector::after {
  content: '';
  position: absolute;
  top: 0; left: -100%; width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, #2ECC71, #5B86E5, transparent);
  transition: left 1.4s 0.2s ease;
}
.connector.lit::after { left: 120%; }

/* Mini-net graphic */
.mini-net {
  display: flex;
  justify-content: center;
  margin-top: 8px;
  opacity: 0.45;
}

/* ════════════════════════════════════════════════════════════════════
   CTA
════════════════════════════════════════════════════════════════════ */
#cta {
  background: radial-gradient(ellipse at 50% 0%, #091630 0%, #04050f 55%);
  padding: 80px 40px;
  text-align: center;
}
.cta-glow {
  position: absolute;
  width: 600px; height: 350px;
  background: radial-gradient(ellipse, rgba(46,204,113,0.11) 0%, transparent 65%);
  top: 50%; left: 50%;
  transform: translate(-50%, -60%);
  pointer-events: none;
  animation: glowPulse 4.5s ease-in-out infinite;
}
@keyframes glowPulse {
  0%,100% { opacity:0.7; transform:translate(-50%,-60%) scale(1); }
  50%      { opacity:1;   transform:translate(-50%,-60%) scale(1.15); }
}
.cta-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 5px;
  text-transform: uppercase;
  color: #2ECC71;
  margin-bottom: 18px;
  position: relative; z-index: 1;
}
.cta-title {
  font-size: clamp(30px, 5vw, 54px);
  font-weight: 900;
  letter-spacing: -1.5px;
  margin-bottom: 16px;
  line-height: 1.1;
  position: relative; z-index: 1;
}
.cta-sub {
  font-size: 15px;
  color: #8892b0;
  margin-bottom: 48px;
  max-width: 400px;
  margin-left: auto; margin-right: auto;
  line-height: 1.7;
  position: relative; z-index: 1;
}
.enter-btn {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  height: 60px;
  padding: 0 44px;
  border-radius: 100px;
  border: none;
  background: linear-gradient(135deg, #2ECC71 0%, #5B86E5 100%);
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.3px;
  cursor: pointer;
  box-shadow: 0 0 0 0 rgba(46,204,113,0.4);
  transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
  position: relative; z-index: 1;
  animation: btnGlow 3s 2s ease-in-out infinite;
}
@keyframes btnGlow {
  0%,100% { box-shadow: 0 0 20px rgba(46,204,113,0.25), 0 0 0px rgba(91,134,229,0.2); }
  50%      { box-shadow: 0 0 40px rgba(46,204,113,0.45), 0 0 20px rgba(91,134,229,0.3); }
}
.enter-btn:hover {
  transform: scale(1.06) translateY(-2px);
  filter: brightness(1.1);
  box-shadow: 0 10px 50px rgba(46,204,113,0.5), 0 4px 20px rgba(91,134,229,0.4) !important;
}
.enter-btn:active { transform: scale(1.01); }
.btn-arrow { font-size: 20px; transition: transform 0.25s ease; }
.enter-btn:hover .btn-arrow { transform: translateX(4px); }
.cta-stack {
  margin-top: 40px;
  display: flex;
  justify-content: center;
  gap: 24px;
  flex-wrap: wrap;
  position: relative; z-index: 1;
}
.stack-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: #3a4466;
}
.stack-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: rgba(91,134,229,0.4);
}
</style>
</head>
<body>

<!-- ══════════════════════════ HERO ══════════════════════════════ -->
<section id="hero" class="full-section">
  <canvas id="heroCanvas"></canvas>
  <div id="heroOverlay"></div>

  <div class="hero-content">
    <span class="hero-eyebrow">Machine Learning Interactivo</span>
    <h1 class="hero-title grad">Neural<br>Decode</h1>
    <span class="hero-rule"></span>
    <p class="hero-sub">
      Entrena tu propia red neuronal.<br>
      Dibuja numeros y letras. Mira como piensa la red en tiempo real.
    </p>
    <div class="hero-pills">
      <span class="pill"><span class="pill-dot"></span>Digitos MNIST</span>
      <span class="pill"><span class="pill-dot"></span>Letras EMNIST A-Z</span>
      <span class="pill"><span class="pill-dot"></span>Visualizaciones en vivo</span>
    </div>
  </div>

  <div class="scroll-indicator">
    <span class="scroll-label">SCROLL</span>
    <svg class="scroll-arrow" width="20" height="11" viewBox="0 0 20 11">
      <path d="M1 1L10 10L19 1" stroke="#5B86E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
</section>

<!-- ══════════════════════ FEATURES ══════════════════════════════ -->
<section id="features" class="full-section">
  <p class="section-label reveal">CAPACIDADES</p>
  <h2 class="section-title reveal d1">Que vas a poder hacer</h2>
  <div class="cards">
    <div class="card reveal d1">
      <span class="card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#5B86E5" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/></svg></span>
      <div class="card-title">Configura tu red</div>
      <div class="card-body">
        Ajusta <strong>capas ocultas, neuronas, learning rate y batch size</strong>
        con sliders. La arquitectura se previsualiza en tiempo real.
      </div>
    </div>
    <div class="card reveal d2">
      <span class="card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="2" stroke-linecap="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg></span>
      <div class="card-title">Dibuja y predeci</div>
      <div class="card-body">
        Traza un digito o letra en el canvas. La red
        <strong>construye su grafo capa a capa</strong> con particulas fluyendo.
        Cada neurona muestra su activacion exacta.
      </div>
    </div>
    <div class="card reveal d3">
      <span class="card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#36D1DC" stroke-width="2" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg></span>
      <div class="card-title">Explora las representaciones</div>
      <div class="card-body">
        Visualiza <strong>mapas de saliencia, t-SNE, matriz de confusion
        y pesos por capa</strong>. Entende exactamente como la red toma decisiones.
      </div>
    </div>
  </div>
</section>

<!-- ══════════════════════ HOW IT WORKS ══════════════════════════ -->
<section id="how" class="full-section">
  <p class="section-label reveal">EL FLUJO</p>
  <h2 class="section-title reveal d1">Como funciona</h2>

  <div class="steps-row">
    <div class="step reveal d1">
      <div class="step-icon-wrap"><svg viewBox="0 0 24 24" fill="none" stroke="#5B86E5" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></div>
      <div class="step-heading">Configuras</div>
      <div class="step-desc">Elegis capas, neuronas<br>y parametros</div>
    </div>
    <div class="connector reveal d1"></div>
    <div class="step reveal d2">
      <div class="step-icon-wrap"><svg viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="2" stroke-linecap="round"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div>
      <div class="step-heading">Entrenas</div>
      <div class="step-desc">La red aprende de<br>60.000+ imagenes</div>
    </div>
    <div class="connector reveal d2"></div>
    <div class="step reveal d3">
      <div class="step-icon-wrap"><svg viewBox="0 0 24 24" fill="none" stroke="#36D1DC" stroke-width="2" stroke-linecap="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg></div>
      <div class="step-heading">Dibujas</div>
      <div class="step-desc">Un numero o letra<br>en el canvas</div>
    </div>
    <div class="connector reveal d3"></div>
    <div class="step reveal d4">
      <div class="step-icon-wrap"><svg viewBox="0 0 24 24" fill="none" stroke="#e6edf3" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg></div>
      <div class="step-heading">Predecis</div>
      <div class="step-desc">La red lo reconoce y<br>muestra su razonamiento</div>
    </div>
  </div>

  <div class="mini-net reveal d2">
    <svg width="360" height="80" viewBox="0 0 360 80">
      <g stroke="#1a2a4a" stroke-width="0.7" opacity="0.9">
        <line x1="30" y1="16" x2="96" y2="9"/><line x1="30" y1="16" x2="96" y2="25"/>
        <line x1="30" y1="32" x2="96" y2="9"/><line x1="30" y1="32" x2="96" y2="25"/>
        <line x1="30" y1="32" x2="96" y2="41"/><line x1="30" y1="48" x2="96" y2="25"/>
        <line x1="30" y1="48" x2="96" y2="41"/><line x1="30" y1="48" x2="96" y2="57"/>
        <line x1="30" y1="64" x2="96" y2="41"/><line x1="30" y1="64" x2="96" y2="57"/>
        <line x1="108" y1="9"  x2="174" y2="16"/><line x1="108" y1="25" x2="174" y2="16"/>
        <line x1="108" y1="25" x2="174" y2="32"/><line x1="108" y1="41" x2="174" y2="32"/>
        <line x1="108" y1="41" x2="174" y2="48"/><line x1="108" y1="57" x2="174" y2="48"/>
        <line x1="108" y1="57" x2="174" y2="64"/><line x1="186" y1="16" x2="252" y2="25"/>
        <line x1="186" y1="32" x2="252" y2="25"/><line x1="186" y1="32" x2="252" y2="41"/>
        <line x1="186" y1="48" x2="252" y2="41"/><line x1="186" y1="48" x2="252" y2="57"/>
        <line x1="186" y1="64" x2="252" y2="57"/><line x1="264" y1="25" x2="330" y2="32"/>
        <line x1="264" y1="41" x2="330" y2="32"/><line x1="264" y1="41" x2="330" y2="48"/>
        <line x1="264" y1="57" x2="330" y2="48"/>
      </g>
      <circle cx="30" cy="16" r="6" fill="#0b1e3a" stroke="#3a7bcc" stroke-width="1.2"/>
      <circle cx="30" cy="32" r="6" fill="#0b1e3a" stroke="#3a7bcc" stroke-width="1.2"/>
      <circle cx="30" cy="48" r="6" fill="#0b1e3a" stroke="#3a7bcc" stroke-width="1.2"/>
      <circle cx="30" cy="64" r="6" fill="#0b1e3a" stroke="#3a7bcc" stroke-width="1.2"/>
      <circle cx="108" cy="9"  r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="108" cy="25" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="108" cy="41" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="108" cy="57" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="186" cy="16" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="186" cy="32" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="186" cy="48" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="186" cy="64" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="264" cy="25" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="264" cy="41" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="264" cy="57" r="6" fill="#0e1623" stroke="#4a6080" stroke-width="1.2"/>
      <circle cx="330" cy="32" r="7" fill="#061810" stroke="#22aa55" stroke-width="1.8"/>
      <circle cx="330" cy="48" r="7" fill="#061810" stroke="#22aa55" stroke-width="1.8"/>
      <circle cx="330" cy="32" r="11" fill="none" stroke="#2ECC71" stroke-width="0.5" opacity="0.35"/>
      <circle cx="330" cy="48" r="11" fill="none" stroke="#2ECC71" stroke-width="0.5" opacity="0.35"/>
    </svg>
  </div>
</section>

<!-- ══════════════════════════ CTA ════════════════════════════════ -->
<section id="cta" class="full-section">
  <div class="cta-glow"></div>
  <p class="cta-label reveal">Empeza ahora</p>
  <h2 class="cta-title reveal d1">Listo para entrenar<br>tu primera red</h2>
  <p class="cta-sub reveal d2">
    Todo corre en tu máquina.<br>
    Sin servidores externos. Sin datos enviados.
  </p>
  <button class="enter-btn reveal d3" id="enterBtn">
    Entrar a la plataforma <span class="btn-arrow">→</span>
  </button>
  <div class="cta-stack reveal d4">
    <span class="stack-item"><span class="stack-dot"></span>PyTorch</span>
    <span class="stack-item"><span class="stack-dot"></span>Streamlit</span>
    <span class="stack-item"><span class="stack-dot"></span>scikit-learn</span>
    <span class="stack-item"><span class="stack-dot"></span>Plotly</span>
  </div>
</section>

<script>
// ════════════════════════════════════════════════════════════════════
// ENTER BUTTON — navigate the top-level page to ?app=1
// window.top.location gives us the actual parent URL regardless of
// which port Streamlit is running on.
// ════════════════════════════════════════════════════════════════════
document.getElementById('enterBtn').addEventListener('click', function() {
  try {
    var top = window.top;
    var dest = top.location.origin + top.location.pathname + '?app=1';
    top.location.href = dest;
  } catch (e) {
    // Fallback: navigate the iframe itself (Streamlit will pick up ?app=1)
    window.location.href = window.location.origin + '/?app=1';
  }
});

// ════════════════════════════════════════════════════════════════════
// FIX: Set section heights using window.innerHeight (not vh units)
// Inside a Streamlit iframe, vh = iframe's full height, not the
// visible viewport. We must use window.innerHeight instead.
// ════════════════════════════════════════════════════════════════════
function setSectionHeights() {
  const h = window.innerHeight;
  document.querySelectorAll('.full-section').forEach(s => {
    s.style.minHeight = h + 'px';
  });
  // CTA can be shorter
  const cta = document.getElementById('cta');
  if (cta) cta.style.minHeight = Math.round(h * 0.75) + 'px';
}
setSectionHeights();
window.addEventListener('resize', setSectionHeights);

// ════════════════════════════════════════════════════════════════════
// INTERSECTION OBSERVER — scroll-driven animations
// ════════════════════════════════════════════════════════════════════
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      if (e.target.classList.contains('step')) e.target.classList.add('visible');
      if (e.target.classList.contains('connector')) {
        setTimeout(() => e.target.classList.add('lit'), 150);
      }
    }
  });
}, { threshold: 0.10 });

document.querySelectorAll('.reveal, .step, .connector').forEach(el => io.observe(el));

// ════════════════════════════════════════════════════════════════════
// HERO CANVAS — autonomous neural network background
// ════════════════════════════════════════════════════════════════════
const canvas = document.getElementById('heroCanvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
  const hero = document.getElementById('hero');
  canvas.width  = hero.offsetWidth;
  canvas.height = hero.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', () => { setSectionHeights(); resizeCanvas(); });

// Network topology: 5 layers, denser center
const LAYER_COUNTS = [5, 8, 11, 8, 5];

// Build node list with animation params
const nodes = [];
let nodeIdx = 0;
LAYER_COUNTS.forEach((count, li) => {
  for (let ni = 0; ni < count; ni++) {
    nodes.push({
      li, ni, count,
      phase:   Math.random() * Math.PI * 2,
      speed:   0.35 + Math.random() * 0.55,
      wobXph:  Math.random() * Math.PI * 2,
      wobYph:  Math.random() * Math.PI * 2,
      wobSpd:  0.25 + Math.random() * 0.35,
    });
    nodeIdx++;
  }
});

// Get node screen position (computed each frame, responsive)
function getPos(n) {
  const W = canvas.width, H = canvas.height;
  const N = LAYER_COUNTS.length;
  const padX = W * 0.10;
  const lx   = padX + (n.li / (N - 1)) * (W - 2 * padX);
  const padY = H * 0.15;
  const gap  = n.count > 1 ? (H - 2 * padY) / (n.count - 1) : 0;
  const ly   = n.count > 1 ? padY + n.ni * gap : H / 2;
  const t    = performance.now() * 0.001;
  const wobAmp = 4;
  return {
    x: lx + Math.sin(t * n.wobSpd + n.wobXph) * wobAmp,
    y: ly + Math.cos(t * n.wobSpd + n.wobYph) * wobAmp,
  };
}

// Build edges: each node connects to all in next layer
const edges = [];
(function() {
  const starts = [];
  let idx = 0;
  LAYER_COUNTS.forEach(c => { starts.push(idx); idx += c; });
  for (let li = 0; li < LAYER_COUNTS.length - 1; li++) {
    for (let ni = 0; ni < LAYER_COUNTS[li]; ni++) {
      for (let nj = 0; nj < LAYER_COUNTS[li + 1]; nj++) {
        edges.push({
          a: starts[li] + ni,
          b: starts[li + 1] + nj,
          w: (Math.random() - 0.5) * 2,
        });
      }
    }
  }
})();

// Particles
const MAX_P = 100;
const parts  = [];
for (let i = 0; i < 50; i++) {
  const e = edges[Math.floor(Math.random() * edges.length)];
  parts.push({ e, t: Math.random(), spd: 0.003 + Math.random() * 0.004 });
}
function spawnP() {
  if (parts.length >= MAX_P) return;
  const e = edges[Math.floor(Math.random() * edges.length)];
  parts.push({ e, t: 0, spd: 0.003 + Math.random() * 0.004 });
}

// ── Render loop ────────────────────────────────────────────────────
let lastPSpawn = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  if (!canvas.width || !canvas.height) return;

  const W = canvas.width, H = canvas.height;
  const t = ts * 0.001;

  // Background gradient
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#04050f');
  bg.addColorStop(1, '#070c1c');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Per-node activation (sine-based)
  const acts = nodes.map(n => 0.3 + 0.7 * Math.abs(Math.sin(t * n.speed + n.phase)));
  const pos  = nodes.map(n => getPos(n));

  // Draw edges
  edges.forEach(e => {
    const pa = pos[e.a], pb = pos[e.b];
    const alpha = acts[e.a] * acts[e.b] * 0.45;
    const col = e.w > 0 ? `rgba(68,102,204,${alpha.toFixed(2)})` : `rgba(200,110,51,${alpha.toFixed(2)})`;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = col;
    ctx.lineWidth = 0.75;
    ctx.stroke();
  });

  // Spawn + draw particles
  if (ts - lastPSpawn > 100) { spawnP(); lastPSpawn = ts; }
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    p.t += p.spd;
    if (p.t > 1) { parts.splice(i, 1); continue; }
    const pa = pos[p.e.a], pb = pos[p.e.b];
    const px = pa.x + (pb.x - pa.x) * p.t;
    const py = pa.y + (pb.y - pa.y) * p.t;
    const a  = Math.sin(p.t * Math.PI) * 0.75;
    ctx.globalAlpha = a;
    ctx.beginPath();
    ctx.arc(px, py, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = p.e.w > 0 ? '#88aaff' : '#ffaa66';
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Draw nodes
  const NR = 7;
  nodes.forEach((n, i) => {
    const { x, y } = pos[i];
    const act = acts[i];
    const isIn  = n.li === 0;
    const isOut = n.li === LAYER_COUNTS.length - 1;
    const gCol  = isIn ? 'rgba(85,170,255,' : isOut ? 'rgba(46,204,113,' : 'rgba(100,140,190,';
    const sCol  = isIn ? '#3a7bcc' : isOut ? '#22aa55' : '#4a6080';
    const fCol  = isIn ? '#0b1e3a' : isOut ? '#061810' : '#0e1623';

    // Outer glow
    const g = ctx.createRadialGradient(x, y, 0, x, y, NR * 3);
    g.addColorStop(0, gCol + (act * 0.28).toFixed(2) + ')');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, NR * 3, 0, Math.PI * 2);
    ctx.fill();

    // Node body
    const r = NR * (0.88 + act * 0.14);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fCol;
    ctx.fill();
    ctx.strokeStyle = sCol;
    ctx.lineWidth = isOut ? 1.7 : 1.3;
    ctx.stroke();
  });
}
requestAnimationFrame(frame);
</script>
</body>
</html>"""
