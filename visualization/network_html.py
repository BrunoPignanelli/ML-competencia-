"""
network_html.py — Componente HTML/Canvas animado de arquitectura de red neuronal.

Al cargar, la red se CONSTRUYE capa por capa:
  - Fase 0: fondo + etiquetas aparecen
  - Fase 1: recuadro de entrada + nodos de input (pop-in con spring)
  - Fase 2: conexiones input→H1 se dibujan de izquierda a derecha
  - Fase 3: nodos H1 aparecen con burst de activación
  - … repite por cada capa …
  - Fase N: nodos de salida + pulse verde en el nodo predicho
  - Fase N+1: partículas animadas comienzan (modo interactivo continuo)

Se embebe con st.components.v1.html().
"""

import json
from typing import List, Optional

import numpy as np
import torch

INPUT_ACTUAL  = 784
OUTPUT_ACTUAL = 10
INPUT_DISPLAY = 10

# ─────────────────────────────────────────────────────────────────────────────
# HTML/JS Template — usa __DATA__ y __HEIGHT__ como placeholders
# ─────────────────────────────────────────────────────────────────────────────
_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; overflow: hidden; font-family: -apple-system, 'Inter', sans-serif; }
#wrap { position: relative; width: 100%; }
canvas { display: block; cursor: crosshair; }
#tip {
  position: fixed;
  background: rgba(5,9,20,0.96);
  color: #c5d8f5;
  padding: 9px 13px;
  border-radius: 10px;
  font-size: 12px;
  line-height: 1.75;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s;
  border: 1px solid rgba(70,120,210,0.45);
  z-index: 9999;
  max-width: 175px;
}
</style>
</head>
<body>
<div id="wrap"><canvas id="c"></canvas><div id="tip"></div></div>
<script>
(function(){

// ── Data ────────────────────────────────────────────────────────────────────
const DATA = __DATA__;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const tip    = document.getElementById('tip');

// ── Palette ──────────────────────────────────────────────────────────────────
const P = {
  bg1:'#060a14', bg2:'#0b1422',
  grid:'rgba(28,48,90,0.16)',
  input:  {f:'#0b1e3a', s:'#3a7bcc', g:'#55aaff'},
  hidden: {f:'#0e1623', s:'#4a6080', g:'#6a90c0'},
  output: {f:'#1a0810', s:'#aa3855', g:'#ff6077'},
  pred:   {f:'#061810', s:'#22aa55', g:'#2ecc71'},
  posE:'#4466cc', negE:'#cc7733',
  posP:'#88aaff', negP:'#ffaa55',
};

// ── Constants ────────────────────────────────────────────────────────────────
const H   = __HEIGHT__;
const MAR = {t:38, r:140, b:52, l:72};
const NR  = 9;
const DPR = Math.min(window.devicePixelRatio||1, 2);

// ── Reveal schedule ──────────────────────────────────────────────────────────
const nL = DATA.layers.length;
const T_BG    = 30;   // ticks for bg/labels fade
const T_LAYER = 38;   // ticks to pop in one layer's nodes
const T_EDGES = 52;   // ticks to draw one inter-layer edge set

const layerStart = [], edgeStart = [];
layerStart[0] = T_BG;
let cur = T_BG + T_LAYER;
for (let i = 0; i < nL-1; i++) {
  edgeStart[i] = cur;
  cur += T_EDGES;
  layerStart[i+1] = cur;
  cur += T_LAYER;
}
const PULSE_START  = cur;          // predicted node pulses after all layers
const REVEAL_TOTAL = cur + 80;     // 80 ticks for 3 pulses, then particles

let revealTick = 0;
let REVEALED   = false;
let tick       = 0;

// ── Layout ──────────────────────────────────────────────────────────────────
let W, plotW, plotH, layerX;
const DPR2 = DPR;

function resize() {
  W = document.getElementById('wrap').clientWidth || 800;
  canvas.width  = W * DPR2;
  canvas.height = H * DPR2;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.setTransform(DPR2,0,0,DPR2,0,0);
  buildLayout();
}

function buildLayout() {
  plotW = W - MAR.l - MAR.r;
  plotH = H - MAR.t  - MAR.b;
  layerX = DATA.layers.map((_,i) => MAR.l + (i/(nL-1)) * plotW);
}

function ny(li, ni) {
  const n = DATA.layers[li];
  return n===1 ? MAR.t+plotH/2 : MAR.t + (ni/(n-1))*plotH;
}

// ── Easing ────────────────────────────────────────────────────────────────────
function cl(t) { return Math.max(0, Math.min(1, t)); }
function easeOut(t)  { return 1 - Math.pow(1-cl(t), 3); }
function easeOutBack(t) {
  t = cl(t);
  const c1=1.70158, c3=c1+1;
  return 1 + c3*Math.pow(t-1,3) + c1*Math.pow(t-1,2);
}
function easeIn(t) { return cl(t)*cl(t); }

function prog(start, dur) { return easeOut(cl((revealTick-start)/dur)); }
function progRaw(start, dur) { return cl((revealTick-start)/dur); }

// ── Activation helpers ───────────────────────────────────────────────────────
function actNorm(li, ni) {
  if (!DATA.activations||!DATA.activations[li]) return 0.5;
  const a = DATA.activations[li];
  const mx = a.reduce((m,v)=>Math.max(m,Math.abs(v)), 1e-9);
  return Math.min(Math.abs(a[ni])/mx, 1);
}
function probNorm(ni) {
  if (!DATA.probs) return 0.5;
  const mx = Math.max(...DATA.probs, 1e-9);
  return DATA.probs[ni]/mx;
}

// ── Particle system (starts after REVEAL_TOTAL) ───────────────────────────────
let parts = [];

function buildParticles() {
  parts = [];
  if (!DATA.weights) return;
  for (let li=0; li<nL-1; li++) {
    const W2 = DATA.weights[li]; if (!W2) continue;
    for (let d=0; d<DATA.layers[li+1]; d++) {
      for (let s=0; s<DATA.layers[li]; s++) {
        const wv=W2[d][s], aw=Math.abs(wv);
        if (aw<0.22) continue;
        const srcA = actNorm(li,s);
        const cnt  = (aw*(0.4+0.6*srcA)) > 0.65 ? 2 : 1;
        for (let k=0; k<cnt; k++)
          parts.push({li,s,d, t:Math.random(), sp:0.0022+aw*0.005, pos:wv>0, aw});
        if (parts.length>300) { li=nL; break; }
      }
    }
  }
}

function drawParticles() {
  for (const p of parts) {
    p.t += p.sp; if (p.t>1) p.t-=1;
    const x1=layerX[p.li], y1=ny(p.li,p.s);
    const x2=layerX[p.li+1], y2=ny(p.li+1,p.d);
    const px=x1+(x2-x1)*p.t, py=y1+(y2-y1)*p.t;
    const fade=Math.sin(p.t*Math.PI);
    const col=p.pos?P.posP:P.negP, r=1.4+p.aw*1.8;
    ctx.beginPath(); ctx.arc(px,py,r*2.8,0,Math.PI*2);
    ctx.fillStyle=ha(col,fade*0.22); ctx.fill();
    ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
    ctx.fillStyle=ha(col,fade*0.92); ctx.fill();
  }
}

// ── Hover ─────────────────────────────────────────────────────────────────────
let hov = null;
canvas.addEventListener('mousemove', e=>{
  const r=canvas.getBoundingClientRect();
  hov=hitTest(e.clientX-r.left, e.clientY-r.top);
  showTip(e.clientX, e.clientY);
});
canvas.addEventListener('mouseleave', ()=>{ hov=null; tip.style.opacity='0'; });

function hitTest(mx,my) {
  for (let li=0; li<nL; li++)
    for (let ni=0; ni<DATA.layers[li]; ni++) {
      const dx=mx-layerX[li], dy=my-ny(li,ni);
      if (dx*dx+dy*dy<(NR+5)*(NR+5)) return {li,ni};
    }
  return null;
}

function showTip(cx,cy) {
  if (!hov) { tip.style.opacity='0'; return; }
  const {li,ni}=hov;
  let html='';
  if (li===0) {
    html=`<b style="color:#6eb3ff">Input · N${ni}</b><br>Total: ${DATA.layerActual[0]} px`;
    if (DATA.activations) html+=`<br>Valor: ${DATA.activations[0][ni].toFixed(3)}`;
  } else if (li===nL-1) {
    const ip=ni===DATA.predicted;
    const lbl=(DATA.classLabels&&DATA.classLabels[ni]!==undefined)?DATA.classLabels[ni]:String(ni);
    html=`<b style="color:${ip?'#2ecc71':'#ff6080'}">${lbl}${ip?' ✓':''}</b>`;
    if (DATA.probs) html+=`<br>Prob: <b>${(DATA.probs[ni]*100).toFixed(2)}%</b>`;
  } else {
    html=`<b style="color:#99bbdd">Hidden L${li} · N${ni}</b>`;
    if (DATA.activations&&DATA.activations[li]) {
      const v=DATA.activations[li][ni];
      html+=`<br>Activación: ${v.toFixed(4)}<br>ReLU: ${v>0?'<span style="color:#2ecc71">activa ●</span>':'<span style="color:#666">inactiva ○</span>'}`;
    }
  }
  tip.innerHTML=html;
  tip.style.opacity='1';
  tip.style.left=(cx+14)+'px';
  tip.style.top=(cy-10)+'px';
}

// ── Helper ────────────────────────────────────────────────────────────────────
function ha(hex,a) {
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

// ── Draw: background ──────────────────────────────────────────────────────────
function drawBg(alpha) {
  const g=ctx.createLinearGradient(0,0,W,H);
  g.addColorStop(0,P.bg1); g.addColorStop(1,P.bg2);
  ctx.globalAlpha=cl(alpha);
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle=P.grid; ctx.lineWidth=0.5;
  for (let x=0;x<W;x+=45){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  for (let y=0;y<H;y+=45){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  ctx.globalAlpha=1;
}

// ── Draw: edges for a single layer pair with reveal progress ─────────────────
function drawEdgesForPair(li, edgeProg, fullAlpha) {
  if (edgeProg<=0) return;
  const W2 = DATA.weights?DATA.weights[li]:null;
  const nSrc=DATA.layers[li], nDst=DATA.layers[li+1];

  for (let d=0;d<nDst;d++) {
    for (let s=0;s<nSrc;s++) {
      const wv=W2?W2[d][s]:0, aw=W2?Math.abs(wv):0.28;
      const x1=layerX[li], y1=ny(li,s);
      const x2=layerX[li+1], y2=ny(li+1,d);

      const sHov=hov&&hov.li===li&&hov.ni===s;
      const dHov=hov&&hov.li===li+1&&hov.ni===d;
      const conn=sHov||dHov;

      let alpha;
      if (!REVEALED||!fullAlpha) {
        alpha=(W2?aw*0.50:0.10)*edgeProg;
      } else if (!hov)      alpha=W2?aw*0.50:0.09;
      else if (conn)        alpha=W2?Math.max(aw,0.55):0.5;
      else                  alpha=W2?aw*0.05:0.02;

      if (alpha<0.015) continue;

      const col=W2?(wv>=0?P.posE:P.negE):'#334477';
      const lw=conn?Math.max(aw*3,1.8):(W2?aw*1.4+0.2:0.35);

      // Draw partial edge based on edgeProg (grows from left to right)
      const ex=x1+(x2-x1)*edgeProg;
      const ey=y1+(y2-y1)*edgeProg;

      ctx.beginPath();
      ctx.moveTo(x1,y1);
      ctx.lineTo(ex,ey);
      ctx.strokeStyle=ha(col,alpha);
      ctx.lineWidth=lw;
      ctx.stroke();
    }
  }
}

// ── Draw: burst ring (expanding + fading) ─────────────────────────────────────
function drawBurst(x, y, col, burstProg) {
  if (burstProg<=0||burstProg>=1) return;
  const r = NR*1.5 + burstProg*28;
  const a = (1-burstProg)*0.65;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
  ctx.strokeStyle=ha(col,a);
  ctx.lineWidth=2.5;
  ctx.stroke();
  // Inner fill flash
  ctx.beginPath(); ctx.arc(x,y,NR*1.2,0,Math.PI*2);
  ctx.fillStyle=ha(col,(1-burstProg)*0.3);
  ctx.fill();
}

// ── Draw: nodes for a single layer with reveal progress ───────────────────────
function drawNodesForLayer(li, nodeProg, burstProg) {
  if (nodeProg<=0) return;
  const isOut=li===nL-1;
  const n=DATA.layers[li];

  for (let ni=0;ni<n;ni++) {
    const x=layerX[li], y=ny(li,ni);
    const isPred=isOut&&ni===DATA.predicted;
    const act=isOut&&DATA.probs?probNorm(ni):actNorm(li,ni);
    const pulse=0.88+0.12*Math.sin(tick*0.04+ni*0.85+li*1.6);
    const isHov=hov&&hov.li===li&&hov.ni===ni;

    const col=isPred?P.pred:li===0?P.input:isOut?P.output:P.hidden;

    // Spring pop-in: scale grows from 0 with overshoot
    const springScale = easeOutBack(nodeProg);
    const alphaScale  = easeOut(nodeProg);
    const r = NR * Math.max(0, springScale);
    if (r<=0) continue;

    const glR = r*(1.6+act*2.8)*pulse;

    // Aura glow
    const aura=ctx.createRadialGradient(x,y,0,x,y,glR*2);
    aura.addColorStop(0, ha(col.g, act*0.38*pulse*alphaScale));
    aura.addColorStop(0.5,ha(col.g, act*0.10*alphaScale));
    aura.addColorStop(1,'transparent');
    ctx.beginPath(); ctx.arc(x,y,glR*2,0,Math.PI*2);
    ctx.fillStyle=aura; ctx.fill();

    // Body
    ctx.globalAlpha=alphaScale;
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle=col.f; ctx.fill();

    // Border
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.strokeStyle=isHov?'#ffffff':ha(col.s,0.65+act*0.35);
    ctx.lineWidth=isHov?2.5:1.5; ctx.stroke();

    // Inner shine
    if (act>0.12) {
      const sh=ctx.createRadialGradient(x-r*0.3,y-r*0.3,0,x,y,r);
      sh.addColorStop(0,ha(col.g,act*0.58));
      sh.addColorStop(1,'transparent');
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=sh; ctx.fill();
    }

    // Hover ring
    if (isHov) {
      ctx.beginPath(); ctx.arc(x,y,r+5.5,0,Math.PI*2);
      ctx.strokeStyle=ha(col.g,0.65);
      ctx.lineWidth=1; ctx.stroke();
    }
    ctx.globalAlpha=1;

    // Burst ring on initial appearance
    if (burstProg>0) drawBurst(x,y,col.g,burstProg);
  }
}

// ── Draw: output probability labels ───────────────────────────────────────────
function drawOutputLabels(labelProg) {
  if (!DATA.probs||labelProg<=0) return;
  const li=nL-1;
  const BAR=54;
  ctx.globalAlpha=easeOut(labelProg);

  for (let ni=0;ni<DATA.layers[li];ni++) {
    const x=layerX[li]+NR+7, y=ny(li,ni);
    const p=DATA.probs[ni], ip=ni===DATA.predicted;
    const c=ip?'#2ecc71':'#4d6a99';

    ctx.textAlign='left';
    ctx.font=ip?'bold 12px monospace':'11px monospace';
    const lbl=(DATA.classLabels&&DATA.classLabels[ni]!==undefined)?DATA.classLabels[ni]:String(ni);
    ctx.fillStyle=c; ctx.fillText(lbl+(ip?' ✓':''),x,y+4);

    const bx=x+24;
    ctx.fillStyle='rgba(255,255,255,0.05)';
    ctx.beginPath();
    if(ctx.roundRect)ctx.roundRect(bx,y-3,BAR,5,2); else ctx.rect(bx,y-3,BAR,5);
    ctx.fill();
    const fw=BAR*p;
    if (fw>0.5) {
      ctx.fillStyle=ip?'#2ecc71':'#3d5caa';
      ctx.beginPath();
      if(ctx.roundRect)ctx.roundRect(bx,y-3,fw,5,2); else ctx.rect(bx,y-3,fw,5);
      ctx.fill();
    }
    ctx.font='10px monospace'; ctx.fillStyle=ip?'#2ecc71':'#3d5080';
    ctx.fillText((p*100).toFixed(1)+'%', bx+BAR+5, y+4);
  }
  ctx.globalAlpha=1;
}

// ── Draw: predicted node pulse (3 expanding rings) ────────────────────────────
function drawPredictedPulse() {
  if (!DATA.predicted && DATA.predicted!==0) return;
  const x=layerX[nL-1], y=ny(nL-1, DATA.predicted);
  for (let p=0;p<3;p++) {
    const pStart=PULSE_START+p*25;
    const pProg=cl((revealTick-pStart)/22);
    if (pProg<=0) continue;
    const r=NR*2+pProg*32;
    const a=(1-pProg)*0.75;
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.strokeStyle=ha('#2ecc71',a);
    ctx.lineWidth=2.5+pProg*1.5; ctx.stroke();
  }
}

// ── Draw: input image box ─────────────────────────────────────────────────────
function drawInputBox(boxProg) {
  if (boxProg<=0) return;
  const BW=44,BH=44;
  const bx=layerX[0]-60-BW/2, by=MAR.t+plotH/2-BH/2;
  ctx.globalAlpha=easeOut(boxProg);

  ctx.shadowBlur=14; ctx.shadowColor='rgba(50,110,200,0.35)';
  ctx.fillStyle='rgba(8,18,36,0.8)';
  ctx.strokeStyle='rgba(50,95,175,0.55)';
  ctx.lineWidth=1.5;
  ctx.beginPath();
  if(ctx.roundRect)ctx.roundRect(bx,by,BW,BH,5); else ctx.rect(bx,by,BW,BH);
  ctx.fill(); ctx.stroke();
  ctx.shadowBlur=0;

  if (DATA.pixelGrid) {
    const g=DATA.pixelGrid, cw=(BW-4)/28, ch=(BH-4)/28;
    for (let r=0;r<28;r++)
      for (let c=0;c<28;c++) {
        const v=g[r][c];
        if (v>0.03){ctx.fillStyle=`rgba(130,195,255,${v.toFixed(2)})`;ctx.fillRect(bx+2+c*cw,by+2+r*ch,cw,ch);}
      }
  } else {
    ctx.textAlign='center'; ctx.font='9px monospace';
    ctx.fillStyle='rgba(70,130,210,0.55)';
    ctx.fillText('28×28',bx+BW/2,by+BH/2-3);
    ctx.fillText('input',bx+BW/2,by+BH/2+8);
  }

  // Fan lines to input nodes
  for (let ni=0;ni<DATA.layers[0];ni++) {
    ctx.beginPath();
    ctx.moveTo(bx+BW, by+BH/2);
    ctx.lineTo(layerX[0]-NR, ny(0,ni));
    ctx.strokeStyle='rgba(50,95,160,0.18)'; ctx.lineWidth=0.6; ctx.stroke();
  }

  ctx.textAlign='center'; ctx.font='9px sans-serif';
  ctx.fillStyle='rgba(70,110,180,0.55)';
  ctx.fillText('Input image',bx+BW/2,by+BH+11);
  ctx.globalAlpha=1;
}

// ── Draw: layer labels ─────────────────────────────────────────────────────────
function drawLabels(alpha) {
  if (alpha<=0) return;
  const labY=MAR.t+plotH+18;
  ctx.globalAlpha=easeOut(alpha);
  for (let li=0;li<nL;li++) {
    const x=layerX[li];
    const c=li===0?'#3a7bcc':li===nL-1?'#aa3855':'#4a6888';
    ctx.textAlign='center';
    ctx.font='bold 11px sans-serif'; ctx.fillStyle=c;
    ctx.fillText(DATA.layerNames[li],x,labY);
    const sh=DATA.layers[li],ac=DATA.layerActual[li];
    ctx.font='9px monospace'; ctx.fillStyle='rgba(70,100,150,0.7)';
    ctx.fillText(sh===ac?`${ac} neuronas`:`${sh} / ${ac}`,x,labY+13);
  }
  ctx.globalAlpha=1;
}

// ── Draw: truncation dots ──────────────────────────────────────────────────────
function drawTrunc() {
  for (let li=0;li<nL;li++) {
    if (DATA.layers[li]<DATA.layerActual[li]) {
      ctx.textAlign='center'; ctx.font='bold 13px sans-serif';
      ctx.fillStyle='rgba(70,100,160,0.5)';
      ctx.fillText('···',layerX[li],ny(li,DATA.layers[li]-1)+NR+8);
    }
  }
}

// ── Draw: legend ───────────────────────────────────────────────────────────────
function drawLegend(alpha) {
  if (alpha<=0) return;
  const items=[
    {t:'dot',  c:P.input.s,  l:'Input'},
    {t:'dot',  c:P.hidden.s, l:'Hidden'},
    {t:'dot',  c:P.output.s, l:'Output'},
    {t:'dot',  c:P.pred.s,   l:'Predicted'},
    {t:'line', c:P.posE,     l:'Pos weight'},
    {t:'line', c:P.negE,     l:'Neg weight'},
    {t:'dot',  c:P.posP,     l:'Signal +'},
    {t:'dot',  c:P.negP,     l:'Signal −'},
  ];
  const iW=82, total=items.length*iW;
  let lx=Math.max(4,(W-total)/2), ly=H-13;
  ctx.globalAlpha=easeOut(alpha); ctx.font='10px sans-serif';
  for (const it of items) {
    if(it.t==='dot'){ctx.beginPath();ctx.arc(lx+5,ly-3,4.5,0,Math.PI*2);ctx.fillStyle=it.c;ctx.fill();ctx.strokeStyle='rgba(255,255,255,0.15)';ctx.lineWidth=1;ctx.stroke();}
    else{ctx.beginPath();ctx.moveTo(lx,ly-3);ctx.lineTo(lx+13,ly-3);ctx.strokeStyle=it.c;ctx.lineWidth=2.5;ctx.stroke();}
    ctx.fillStyle='rgba(130,155,200,0.7)'; ctx.textAlign='left';
    ctx.fillText(it.l,lx+14,ly+1); lx+=iW;
  }
  ctx.globalAlpha=1;
}

// ── Main animation loop ────────────────────────────────────────────────────────
function frame() {
  tick++;
  if (!REVEALED) {
    revealTick++;
    if (revealTick>=REVEAL_TOTAL) { REVEALED=true; buildParticles(); }
  }

  ctx.clearRect(0,0,W,H);

  // Background (fades in during phase 0)
  const bgProg = prog(0, T_BG);
  if (bgProg>0) {
    const g=ctx.createLinearGradient(0,0,W,H);
    g.addColorStop(0,P.bg1); g.addColorStop(1,P.bg2);
    ctx.globalAlpha=bgProg; ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle=P.grid; ctx.lineWidth=0.5;
    for(let x=0;x<W;x+=45){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=45){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.globalAlpha=1;
  }

  // Edges (draw per layer pair as they're revealed)
  const fullEdgeAlpha = REVEALED;
  for (let i=0;i<nL-1;i++) {
    const ep = REVEALED ? 1 : progRaw(edgeStart[i], T_EDGES);
    drawEdgesForPair(i, ep, fullEdgeAlpha);
  }

  // Particles (only after full reveal)
  if (REVEALED) drawParticles();

  // Input box (appears with input layer)
  drawInputBox(prog(layerStart[0]-8, T_LAYER+8));

  // Nodes per layer
  for (let li=0;li<nL;li++) {
    const npRaw = progRaw(layerStart[li], T_LAYER);
    const np    = cl(npRaw);
    // Burst progress: peaks at npRaw≈1, fades out
    const burstProg = cl((npRaw-0.85)/0.55);
    drawNodesForLayer(li, np, burstProg);
  }

  // Output labels
  drawOutputLabels(prog(layerStart[nL-1]+T_LAYER*0.3, T_LAYER*0.7));

  // Predicted node pulses (after all revealed)
  drawPredictedPulse();

  // Labels + legend
  drawLabels(bgProg);
  drawTrunc();
  drawLegend(bgProg);

  // Hint
  if (REVEALED) {
    ctx.textAlign='right'; ctx.font='italic 10px sans-serif';
    ctx.fillStyle='rgba(80,110,170,0.5)';
    ctx.fillText('Brillo = fuerza de activación', W-6, H-4);
  }

  requestAnimationFrame(frame);
}

// ── Init ──────────────────────────────────────────────────────────────────────
resize();
frame();

window.addEventListener('resize', ()=>{
  ctx.setTransform(1,0,0,1,0,0);
  resize();
});

})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _sample_indices(actual: int, display: int) -> np.ndarray:
    """Muestrea `display` índices equiespaciados de [0, actual)."""
    if actual <= display:
        return np.arange(actual)
    return np.round(np.linspace(0, actual - 1, display)).astype(int)


def render_network_html(
    hidden_layers_config: List[int],
    weights_list: Optional[List[torch.Tensor]] = None,
    activations: Optional[List[np.ndarray]] = None,
    output_probs: Optional[np.ndarray] = None,
    max_neurons_display: int = 12,
    predicted_class: Optional[int] = None,
    input_image: Optional[np.ndarray] = None,
    height: int = 530,
    output_size: int = 10,
    class_labels: Optional[List[str]] = None,
) -> str:
    """
    Genera el HTML del componente animado de arquitectura de red neuronal.

    La red se construye capa a capa al cargar el componente.
    Pasar datos nuevos (tras cada Predict) reinicia la animación desde cero.

    Parámetros
    ----------
    hidden_layers_config : List[int]
        Neuronas por capa oculta, ej. [128, 64].
    weights_list : list de Tensor, opcional
        Un tensor por inter-capa (n_hidden+1). Shape (out, in).
        De net.get_all_weights().
    activations : list de ndarray, opcional
        De net.forward_with_activations(): [input_784, hidden_1, ..., logits_n].
    output_probs : ndarray shape (output_size,), opcional
        Probabilidades softmax para los nodos de salida.
    max_neurons_display : int
        Máximo de nodos a mostrar en capas ocultas.
    predicted_class : int, opcional
        Índice de la clase predicha (nodo verde con pulsos).
    input_image : ndarray 28×28, opcional
        En rango [0,255] o [0,1] — se muestra en el recuadro de entrada.
    height : int
        Altura en píxeles del iframe.
    output_size : int
        Número de clases de salida (10 para dígitos, 26 para letras).
    class_labels : List[str], opcional
        Etiquetas para cada clase de salida. Por defecto ["0"…"9"] o según output_size.
    """
    if class_labels is None:
        class_labels = [str(i) for i in range(output_size)]

    all_actual  = [INPUT_ACTUAL] + list(hidden_layers_config) + [output_size]
    all_display = [INPUT_DISPLAY] + [min(n, max_neurons_display) for n in hidden_layers_config] + [output_size]
    n_layers    = len(all_actual)
    layer_idx   = [_sample_indices(a, d) for a, d in zip(all_actual, all_display)]

    layer_names = ["Input Layer"]
    for i in range(len(hidden_layers_config)):
        layer_names.append(f"Hidden Layer {i+1}")
    layer_names.append("Output Layer")

    # Weights: normalized sub-matrices
    weights_json = None
    if weights_list is not None:
        weights_json = []
        for li in range(n_layers - 1):
            W = weights_list[li].cpu().numpy()
            si, di = layer_idx[li], layer_idx[li + 1]
            W_sub = W[np.ix_(di, si)]
            w_max = np.abs(W_sub).max() + 1e-8
            weights_json.append((W_sub / w_max).tolist())

    # Activations: sampled to displayed indices
    acts_json = None
    if activations is not None:
        acts_json = []
        for li in range(n_layers):
            if li < len(activations):
                acts_json.append(activations[li][layer_idx[li]].tolist())
            else:
                acts_json.append([0.5] * all_display[li])

    # Pixel grid: normalize to [0,1]
    pixel_grid = None
    if input_image is not None:
        img = np.array(input_image, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        pixel_grid = img.tolist()

    data = {
        "layers":       all_display,
        "layerActual":  all_actual,
        "layerNames":   layer_names,
        "weights":      weights_json,
        "activations":  acts_json,
        "probs":        output_probs.tolist() if output_probs is not None else None,
        "predicted":    int(predicted_class) if predicted_class is not None else None,
        "pixelGrid":    pixel_grid,
        "classLabels":  class_labels,
    }

    return (
        _TEMPLATE
        .replace("__DATA__",   json.dumps(data))
        .replace("__HEIGHT__", str(height))
    )
