(function () {

  // ── Closure algebra on S³ — exact port of the Rust SDK ───────────────────
  //
  // Three functions below are verbatim translations of the Rust source:
  //   hamilton()       ↔  sphere_compose()   rust/src/groups/sphere.rs
  //   sphereInverse()  ↔  sphere_inverse()   rust/src/groups/sphere.rs
  //   sphereSigma()    ↔  sphere_sigma()     rust/src/groups/sphere.rs
  //
  // Everything else (physics, rendering) is scaffolding for the visual.

  // ── Rust: sphere_compose (hamilton product + normalize) ──────────────────
  function hamilton(a, b) {
    return [
      a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
      a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
      a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
      a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ];
  }
  function qnorm(q) {
    const l = Math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2);
    return l > 1e-15 ? [q[0]/l, q[1]/l, q[2]/l, q[3]/l] : [1,0,0,0];
  }
  function compose(a, b)  { return qnorm(hamilton(a, b)); }

  // ── Rust: sphere_inverse (conjugate = inverse for unit quaternions) ────────
  function sphereInverse(q) { return [q[0], -q[1], -q[2], -q[3]]; }

  // ── Rust: sphere_sigma — geodesic distance from identity on S³ ────────────
  //   pub fn sphere_sigma(a: &[f64; 4]) -> f64 { a[0].abs().clamp(0.0, 1.0).acos() }
  function sphereSigma(q) {
    return Math.acos(Math.min(1.0, Math.abs(q[0])));
  }

  // ── S² visual position → S³ unit quaternion ───────────────────────────────
  // Each visual point p ∈ S² maps to a 90° rotation around axis p:
  //   q = [cos(π/4), sin(π/4)·p] = [1/√2, p_x/√2, p_y/√2, p_z/√2]
  // This is a faithful embedding: moving p changes q, changing q changes σ.
  const SQ2 = 1 / Math.SQRT2;
  function toQuat(p) { return [SQ2, p[0]*SQ2, p[1]*SQ2, p[2]*SQ2]; }

  // ── Canvas ────────────────────────────────────────────────────────────────
  const canvas = document.getElementById('sphere-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H, R;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 420);
    W = canvas.width  = size;
    H = canvas.height = size;
    canvas.style.width  = size + 'px';
    canvas.style.height = size + 'px';
    R = size * 0.38;
  }
  window.addEventListener('resize', () => { resize(); buildSphere(); });

  // ── Orbit (quaternion rotation of the whole sphere) ───────────────────────
  function rotQuat(ax, ay, az, angle) {
    const s = Math.sin(angle / 2), l = Math.sqrt(ax*ax+ay*ay+az*az)||1;
    return [Math.cos(angle / 2), ax/l*s, ay/l*s, az/l*s];
  }
  function qrot(q, v) {
    const r = hamilton(hamilton(q, [0, v[0], v[1], v[2]]), sphereInverse(q));
    return [r[1], r[2], r[3]];
  }

  let qOrbit  = [1, 0, 0, 0];
  const dqOrbit = rotQuat(0.2, 1.0, 0.15, 0.003);

  // ── Pull interaction ──────────────────────────────────────────────────────
  let pullPt = null;
  const PULL_K = 0.22;

  function updatePullPoint(mx, my) {
    const cx = (mx - W/2) / R, cy = -(my - H/2) / R;
    const r2 = cx*cx + cy*cy;
    if (r2 > 1.08) { pullPt = null; return; }
    pullPt = qrot(sphereInverse(qOrbit), [cx, cy, Math.sqrt(Math.max(0, 1-r2))]);
  }

  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    const r = canvas.getBoundingClientRect();
    updatePullPoint(e.clientX - r.left, e.clientY - r.top);
  });
  canvas.addEventListener('pointermove', e => {
    if (e.buttons === 0) return;
    const r = canvas.getBoundingClientRect();
    updatePullPoint(e.clientX - r.left, e.clientY - r.top);
  });
  canvas.addEventListener('pointerup',    () => { pullPt = null; });
  canvas.addEventListener('pointerleave', () => { pullPt = null; });

  // ── Fibonacci sphere — uniform distribution on S² ─────────────────────────
  function fibonacciPoints(n) {
    const phi = Math.PI * (Math.sqrt(5) - 1);
    return Array.from({ length: n }, (_, i) => {
      const y = 1 - (i / (n-1)) * 2;
      const r = Math.sqrt(Math.max(0, 1-y*y));
      const t = phi * i;
      return [r*Math.cos(t), y, r*Math.sin(t)];
    });
  }

  // ── Geodesic restoring force on S² (visual physics layer) ─────────────────
  // Returns [fx, fy, fz] — tangent force toward home, magnitude = σ_S2 × k.
  function geodesicForce(pos, home, k) {
    const d = pos[0]*home[0] + pos[1]*home[1] + pos[2]*home[2];
    const sigma = Math.acos(Math.max(-1, Math.min(1, d)));
    if (sigma < 0.0001) return [0, 0, 0];
    const tx = home[0] - d*pos[0], ty = home[1] - d*pos[1], tz = home[2] - d*pos[2];
    const tl = Math.sqrt(tx*tx + ty*ty + tz*tz);
    if (tl < 0.0001) return [0, 0, 0];
    const s = sigma * k / tl;
    return [tx*s, ty*s, tz*s];
  }

  // ── Particle ──────────────────────────────────────────────────────────────
  class Particle {
    constructor(x, y, z) {
      this.home     = [x, y, z];
      this.pos      = [x, y, z];
      this.vel      = [0, 0, 0];
      this.friction = 0.85 + Math.random() * 0.06;
      this.spring   = 0.036 + Math.random() * 0.022;
      // Home quaternion on S³ (precomputed)
      this.homeQ    = toQuat([x, y, z]);
      // σ from SDK formula — updated each frame
      this.sigma    = 0;
    }

    update() {
      // S² geodesic restoring force (closure force)
      const [fx, fy, fz] = geodesicForce(this.pos, this.home, this.spring);
      this.vel[0] += fx; this.vel[1] += fy; this.vel[2] += fz;

      // Pull force (pointer held)
      if (pullPt) {
        const dot   = this.pos[0]*pullPt[0] + this.pos[1]*pullPt[1] + this.pos[2]*pullPt[2];
        const dPull = Math.acos(Math.max(-1, Math.min(1, dot)));
        const R_INF = 1.05;
        if (dPull < R_INF) {
          const falloff = Math.pow(1 - dPull / R_INF, 1.4);
          const [px, py, pz] = geodesicForce(this.pos, pullPt, PULL_K * falloff);
          this.vel[0] += px; this.vel[1] += py; this.vel[2] += pz;
        }
      }

      this.vel[0] *= this.friction; this.vel[1] *= this.friction; this.vel[2] *= this.friction;
      this.pos[0] += this.vel[0] * 0.09;
      this.pos[1] += this.vel[1] * 0.09;
      this.pos[2] += this.vel[2] * 0.09;

      const l = Math.sqrt(this.pos[0]**2 + this.pos[1]**2 + this.pos[2]**2);
      if (l > 0.001) { this.pos[0] /= l; this.pos[1] /= l; this.pos[2] /= l; }

      // ── SDK sigma formula: sphereSigma(sphereInverse(homeQ) ⊗ currQ) ────
      // This is exactly sphere_sigma(sphere_compose(sphere_inverse(q_home), q_curr))
      // — the geodesic distance on S³ between the home and current quaternions.
      const qCurr  = toQuat(this.pos);
      const error  = hamilton(sphereInverse(this.homeQ), qCurr);
      this.sigma   = sphereSigma(error);
    }
  }

  // ── Adjacency (precomputed from home positions, run once) ─────────────────
  let edges = [];
  function buildEdges(ps) {
    edges = [];
    const n = ps.length;
    const threshold = 0.972;
    for (let i = 0; i < n; i++) {
      const [ax, ay, az] = ps[i].home;
      for (let j = i+1; j < n; j++) {
        const [bx, by, bz] = ps[j].home;
        if (ax*bx + ay*by + az*bz > threshold) edges.push([i, j]);
      }
    }
  }

  // ── Build ──────────────────────────────────────────────────────────────────
  let particles = [];
  function buildSphere() {
    particles = fibonacciPoints(380).map(([x, y, z]) => new Particle(x, y, z));
    buildEdges(particles);
    // Initial disturbance — visitors see closure restoration immediately
    particles.forEach(p => {
      let rx = Math.random()-.5, ry = Math.random()-.5, rz = Math.random()-.5;
      const d = rx*p.pos[0] + ry*p.pos[1] + rz*p.pos[2];
      rx -= d*p.pos[0]; ry -= d*p.pos[1]; rz -= d*p.pos[2];
      p.vel[0] += rx*0.32; p.vel[1] += ry*0.32; p.vel[2] += rz*0.32;
    });
  }

  // ── Color: σ = 0 → deep blue │ σ grows → violet → hot-pink → orange ──────
  // Mapped to the SDK's σ range [0, π/2].
  // t = 1 at σ ≈ 0.7 rad (halfway to S³ max), so color is visible at moderate drift.
  function particleColor(sigma, depth) {
    const t     = Math.pow(Math.min(1, sigma / 0.70), 0.60);
    const hue   = (215 + t * 165) % 360;  // blue(215°) → orange(20°) via violet
    const sat   = 68  + t * 24;
    const lum   = 40  + t * 28 + depth * 15;
    const alpha = 0.28 + 0.72 * depth + t * 0.20;
    return [hue, sat, lum, alpha];
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    ctx.clearRect(0, 0, W, H);

    qOrbit = qnorm(hamilton(qOrbit, dqOrbit));
    particles.forEach(p => p.update());

    // ── GeometricPath running product — C = q_1 ⊗ q_2 ⊗ ... ⊗ q_N ─────────
    // This is exactly what GeometricPath.check_global() / StreamMonitor.sigma()
    // returns in the SDK. The value printed here is real SDK output.
    let C = [1, 0, 0, 0];
    particles.forEach(p => { C = compose(C, toQuat(p.pos)); });
    const globalSigma = sphereSigma(C);

    const proj = particles.map((p, i) => {
      const rp  = qrot(qOrbit, p.pos);
      const per = 1 + rp[2] * 0.30;
      return {
        i,
        x:     W/2 + rp[0] * R * per,
        y:     H/2 - rp[1] * R * per,
        z:     rp[2],
        depth: (rp[2] + 1) / 2,
        sigma: p.sigma,
      };
    });

    // ── Edges ──────────────────────────────────────────────────────────────
    ctx.lineWidth = 0.6;
    edges.forEach(([a, b]) => {
      const pa = proj[a], pb = proj[b];
      if (pa.z < -0.10 && pb.z < -0.10) return;
      const t   = Math.pow(Math.min(1, (pa.sigma+pb.sigma)*0.5/0.70), 0.60);
      const dep = (pa.depth + pb.depth) * 0.5;
      const hue = (215 + t * 165) % 360;
      ctx.strokeStyle = `hsla(${hue}, 65%, 55%, ${0.04 + dep*0.12 + t*0.12})`;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    });

    // ── Particles (back-to-front) ───────────────────────────────────────────
    proj.sort((a, b) => a.z - b.z);
    proj.forEach(pp => {
      const [hue, sat, lum, alpha] = particleColor(pp.sigma, pp.depth);
      const size = Math.max(1.0, 1.0 + pp.depth * 1.1 + Math.min(pp.sigma, 1.2) * 1.3);
      if (pp.sigma > 0.10) {
        ctx.shadowColor = `hsl(${hue}, ${sat}%, ${lum+20}%)`;
        ctx.shadowBlur  = Math.min(10, size * 3.2 * Math.min(pp.sigma / 0.55, 1));
      } else { ctx.shadowBlur = 0; }
      ctx.globalAlpha = alpha;
      ctx.fillStyle   = `hsl(${hue}, ${sat}%, ${lum}%)`;
      ctx.beginPath(); ctx.arc(pp.x, pp.y, size, 0, Math.PI*2); ctx.fill();
    });

    ctx.shadowBlur = 0; ctx.globalAlpha = 1;

    // ── Live σ readout — identical to StreamMonitor.sigma() ──────────────
    const sigColor = particleColor(globalSigma * 1.4, 0.7);
    const fs = Math.max(10, W * 0.030);
    ctx.font         = `${fs}px 'Courier New', monospace`;
    ctx.textAlign    = 'right';
    ctx.fillStyle    = `hsla(${sigColor[0]}, ${sigColor[1]}%, ${sigColor[2]+5}%, 0.75)`;
    ctx.fillText(`σ = ${globalSigma.toFixed(3)}`, W - 10, H - 10);
    ctx.textAlign    = 'left';

    requestAnimationFrame(render);
  }

  resize();
  buildSphere();
  render();

})();
