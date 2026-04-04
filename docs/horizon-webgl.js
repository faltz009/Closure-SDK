// horizon-webgl.js — Closure geometry: particle morph
// 2D canvas particle system. Each scene = 3D shape projected orthographically.
// Scroll triggers explosion + spring reassembly into next shape.

(function () {

const canvas = document.getElementById("webgl-canvas");
const ctx    = canvas.getContext("2d");
if (!ctx) return;

// ── Config ─────────────────────────────────────────────────────────────────
const NP      = 420;     // particle count
const SPRING  = 0.092;   // spring force toward target
const DAMPEN  = 0.83;    // velocity damping per frame
const EXPLODE = 7.0;     // explosion burst speed

// ── Particle state (typed arrays for speed) ────────────────────────────────
const px = new Float32Array(NP), py = new Float32Array(NP);
const vx = new Float32Array(NP), vy = new Float32Array(NP);
const pz = new Float32Array(NP);            // depth after rotation (for opacity)
const bx = new Float32Array(NP), by = new Float32Array(NP), bz = new Float32Array(NP);

let W = 440, H = 440, CX = 220, CY = 220, R = 150;

// ── Shape generators — flat [x0,y0,z0, x1,y1,z1, …] in ~[-1,1]³ ──────────

function fibSphere(n) {
  const out = [], g = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const t = g * i;
    out.push(r * Math.cos(t), y, r * Math.sin(t));
  }
  return out;
}



// S1: Diamond cubic lattice — FCC basis + 2-atom motif
function diamondPts(n) {
  const fcc = [[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]];
  const basis = [[0,0,0],[.25,.25,.25]];
  const pts = [], s = 5;
  for (let ix=-s; ix<=s; ix++)
    for (let iy=-s; iy<=s; iy++)
      for (let iz=-s; iz<=s; iz++)
        for (const f of fcc)
          for (const b of basis) {
            const x=ix+f[0]+b[0], y=iy+f[1]+b[1], z=iz+f[2]+b[2];
            if (Math.abs(x)<=2.6&&Math.abs(y)<=2.6&&Math.abs(z)<=2.6)
              pts.push(x/2.6*0.88, y/2.6*0.88, z/2.6*0.88);
          }
  // shuffle so subsampling is spatially uniform, not biased to one corner
  for (let i = pts.length/3 - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const a = i*3, b = j*3;
    [pts[a],pts[a+1],pts[a+2], pts[b],pts[b+1],pts[b+2]] =
    [pts[b],pts[b+1],pts[b+2], pts[a],pts[a+1],pts[a+2]];
  }
  const out = [], step = Math.max(1, Math.floor(pts.length/3/n));
  for (let i = 0; i < pts.length && out.length/3 < n; i += 3*step)
    out.push(pts[i], pts[i+1], pts[i+2]);
  return out;
}

// S2: (3,5) Torus knot — star-cross wrapping pattern
function torusKnotPts(n) {
  const p=3, q=5, Rm=0.62, rt=0.26;
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = (i/n)*Math.PI*2;
    const r = Rm + rt*Math.cos(q*t);
    out.push(r*Math.cos(p*t)*1.05, rt*Math.sin(q*t)*1.55, r*Math.sin(p*t)*1.05);
  }
  return out;
}

// S2: DNA double helix
function helixPts(n) {
  const out = [], h2 = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const s  = (i % h2) / Math.max(1, h2 - 1);
    const a  = s * Math.PI * 5;
    const os = i < h2 ? 0 : Math.PI;
    out.push(0.56 * Math.cos(a + os), (s * 2 - 1) * 0.90, 0.56 * Math.sin(a + os));
  }
  return out;
}

// S3: Two orbiting spheres
function orbitPts(n) {
  const h = Math.floor(n / 2);
  const sa = fibSphere(h), sb = fibSphere(n - h);
  const out = [];
  for (let i = 0; i < sa.length; i += 3) out.push(sa[i]*.46+.54, sa[i+1]*.46, sa[i+2]*.46);
  for (let i = 0; i < sb.length; i += 3) out.push(sb[i]*.46-.54, sb[i+1]*.46, sb[i+2]*.46);
  return out;
}

// S4: 3D Lissajous — a=3 b=4 c=5, fills a cube with interleaving paths
function lissajousPts(n) {
  const a=3, b=4, c=5, δ1=Math.PI/4, δ2=Math.PI/3;
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = (i/n)*Math.PI*2;
    out.push(Math.sin(a*t+δ1)*0.88, Math.sin(b*t)*0.88, Math.sin(c*t+δ2)*0.88);
  }
  return out;
}

// S6: Three (3,5) torus knots on orthogonal axes — each wraps around Y, X, Z respectively,
//     threading through each other like a Hopf-fibration visual
function multiTorusKnotPts(n) {
  const p = 3, q = 5, Rm = 0.62, rt = 0.26;
  const out = [];
  const ppc = Math.ceil(n / 3);
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < ppc && out.length / 3 < n; i++) {
      const t = (i / ppc) * Math.PI * 2;
      const r = Rm + rt * Math.cos(q * t);
      const x = r * Math.cos(p * t) * 1.05;
      const y = rt * Math.sin(q * t) * 1.55;
      const z = r * Math.sin(p * t) * 1.05;
      if      (k === 0) out.push(x, y, z);   // big circle in XZ, wraps around Y
      else if (k === 1) out.push(y, z, x);   // big circle in YZ, wraps around X
      else              out.push(z, x, y);   // big circle in XY, wraps around Z
    }
  }
  return out;
}

// S5: Lorenz attractor — chaotic trajectory, butterfly twin lobes
function lorenzPts(n) {
  let x = 0.1, y = 0, z = 0;
  const σ=10, ρ=28, β=8/3, dt=0.005;
  for (let i = 0; i < 2000; i++) {
    const dx=σ*(y-x), dy=x*(ρ-z)-y, dz=x*y-β*z;
    x+=dx*dt; y+=dy*dt; z+=dz*dt;
  }
  const out = [];
  for (let i = 0; out.length/3 < n; i++) {
    const dx=σ*(y-x), dy=x*(ρ-z)-y, dz=x*y-β*z;
    x+=dx*dt; y+=dy*dt; z+=dz*dt;
    if (i%4===0) out.push(x/20*1.05, (z-25)/25*1.05, y/25*1.05);
  }
  return out;
}

function getShape(s) {
  if (s === 0) return diamondPts(NP);       // Intelligence
  if (s === 1) return torusKnotPts(NP);     // Cybersecurity
  if (s === 2) return orbitPts(NP);         // Trusted Communication
  if (s === 3) return lissajousPts(NP);     // Hardware
  if (s === 4) return helixPts(NP);         // Biological Computing
  if (s === 5) return lorenzPts(NP);        // Quantum
  return multiTorusKnotPts(NP);             // Coordination
}

// ── Scene transition ───────────────────────────────────────────────────────
function setScene(s, explode) {
  const shape = getShape(s);
  for (let i = 0; i < NP; i++) {
    bx[i] = shape[i * 3];
    by[i] = shape[i * 3 + 1];
    bz[i] = shape[i * 3 + 2];
  }
  if (explode > 0)
    for (let i = 0; i < NP; i++) {
      const a = Math.random() * Math.PI * 2;
      vx[i] += Math.cos(a) * explode;
      vy[i] += Math.sin(a) * explode;
    }
}

// ── Scroll sync ────────────────────────────────────────────────────────────
const dots       = document.querySelectorAll(".scene-dot");
const slides     = document.querySelectorAll(".hz-slide");
const slidesRoot = document.getElementById("hz-slides");
const NS         = 7;
let tgt = 0, smooth = 0, lastSI = 0;

const io = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting)
      tgt = parseInt(e.target.dataset.scene) / (NS - 1);
  });
}, { root: slidesRoot, threshold: 0.5 });
slides.forEach(s => io.observe(s));

const updateDots = s => {
  const si = Math.min(NS - 1, Math.floor(s * NS));
  dots.forEach((d, i) => d.classList.toggle("active", i === si));
};

// ── Camera drag ────────────────────────────────────────────────────────────
let camYaw = 0, camPitch = 0;
let dragX = 0, dragY = 0, isDragging = false;
let autoRotate = true;

canvas.style.cursor = "grab";
canvas.addEventListener("pointerdown", e => {
  if (autoRotate) {
    // absorb current auto-rotation angle into camYaw/camPitch so view doesn't jump
    const el = (performance.now() - t0) / 1000;
    camYaw   += el * 0.23;
    camPitch += el * 0.11;
    autoRotate = false;
  }
  isDragging = true;
  dragX = e.clientX; dragY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = "grabbing";
});
canvas.addEventListener("pointermove", e => {
  if (!isDragging) return;
  camYaw   += (e.clientX - dragX) * 0.012;
  camPitch += (e.clientY - dragY) * 0.012;
  camPitch  = Math.max(-1.4, Math.min(1.4, camPitch));
  dragX = e.clientX; dragY = e.clientY;
});
canvas.addEventListener("pointerup",    () => { isDragging = false; canvas.style.cursor = "grab"; });
canvas.addEventListener("pointerleave", () => { isDragging = false; canvas.style.cursor = "grab"; });

// ── Resize ─────────────────────────────────────────────────────────────────
const resize = () => {
  const dpr  = Math.min(devicePixelRatio || 1, 1.5);
  const rect = canvas.getBoundingClientRect();
  W = Math.round((rect.width  || 440) * dpr);
  H = Math.round((rect.height || 440) * dpr);
  canvas.width  = W;
  canvas.height = H;
  CX = W / 2;
  CY = H / 2;
  R  = Math.min(W, H) * 0.37;
};
resize();
window.addEventListener("resize", resize);

// ── Init: place particles at scene 0 positions ────────────────────────────
setScene(0, 0);
for (let i = 0; i < NP; i++) {
  px[i] = CX + bx[i] * R;
  py[i] = CY - by[i] * R;
  pz[i] = bz[i];
}

// ── Render loop ────────────────────────────────────────────────────────────
const t0 = performance.now();
let lastNow = t0;

function frame(now) {
  requestAnimationFrame(frame);

  const elapsed = (now - t0) / 1000;
  const dt      = Math.min((now - lastNow) / 1000, 0.05);
  lastNow = now;

  // Smooth scene tracking
  smooth += (tgt - smooth) * (1 - Math.exp(-dt * 10));
  const sceneIdx = Math.round(smooth * (NS - 1));
  updateDots(smooth);

  if (sceneIdx !== lastSI) {
    lastSI = sceneIdx;
    setScene(sceneIdx, EXPLODE);
  }

  // Rotation (precompute sin/cos once)
  const yaw   = autoRotate ? elapsed * 0.23 + camYaw   : camYaw;
  const pitch = autoRotate ? elapsed * 0.11 + camPitch : camPitch;
  const cosA = Math.cos(yaw),   sinA = Math.sin(yaw);
  const cosB = Math.cos(pitch), sinB = Math.sin(pitch);

  // Update each particle toward rotating target
  for (let i = 0; i < NP; i++) {
    const x0 = bx[i], y0 = by[i], z0 = bz[i];
    // rotY
    const x1 =  x0 * cosA + z0 * sinA;
    const z1 = -x0 * sinA + z0 * cosA;
    // rotX
    const y2 = y0 * cosB - z1 * sinB;
    const z2 = y0 * sinB + z1 * cosB;

    const txp = CX + x1 * R;
    const typ = CY - y2 * R;

    vx[i] += (txp - px[i]) * SPRING;
    vy[i] += (typ - py[i]) * SPRING;
    vx[i] *= DAMPEN;
    vy[i] *= DAMPEN;
    px[i] += vx[i];
    py[i] += vy[i];
    pz[i]  = z2;
  }

  // ── Draw ──────────────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, W, H);

  const LINK2 = (R * 0.28) * (R * 0.28);

  // Lines — two depth buckets, two batched paths (avoids per-line state changes)
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.strokeStyle = "rgba(120,120,120,0.10)";
  for (let i = 0; i < NP; i++)
    for (let j = i + 1; j < NP; j++) {
      const dx = px[i] - px[j], dy = py[i] - py[j];
      if (dx * dx + dy * dy < LINK2) {
        const avgD = ((pz[i] + 1) + (pz[j] + 1)) * 0.25;
        if (avgD < 0.5) { ctx.moveTo(px[i], py[i]); ctx.lineTo(px[j], py[j]); }
      }
    }
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "rgba(120,120,120,0.22)";
  for (let i = 0; i < NP; i++)
    for (let j = i + 1; j < NP; j++) {
      const dx = px[i] - px[j], dy = py[i] - py[j];
      if (dx * dx + dy * dy < LINK2) {
        const avgD = ((pz[i] + 1) + (pz[j] + 1)) * 0.25;
        if (avgD >= 0.5) { ctx.moveTo(px[i], py[i]); ctx.lineTo(px[j], py[j]); }
      }
    }
  ctx.stroke();

  // Dots — near-invisible in back, bright silver in front
  for (let i = 0; i < NP; i++) {
    const d     = (pz[i] + 1) * 0.5;
    const alpha = 0.06 + d * 0.78;
    const size  = 0.6 + d * 2.0;
    const v     = Math.round(130 + d * 80); // 130 (back) → 210 (front)
    ctx.fillStyle = `rgba(${v},${v},${v},${alpha.toFixed(2)})`;
    ctx.beginPath();
    ctx.arc(px[i], py[i], size, 0, Math.PI * 2);
    ctx.fill();
  }
}
requestAnimationFrame(frame);

})();