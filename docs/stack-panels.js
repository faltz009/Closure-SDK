(function () {

  // ── Shared closure algebra (exact port of Rust SDK) ─────────────────────────
  function hamilton(a, b) {
    return [
      a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
      a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
      a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
      a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
    ];
  }
  function qnorm(q) {
    const l = Math.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2);
    return l > 1e-15 ? [q[0]/l,q[1]/l,q[2]/l,q[3]/l] : [1,0,0,0];
  }
  function compose(a, b)    { return qnorm(hamilton(a, b)); }
  function sphereInverse(q) { return [q[0], -q[1], -q[2], -q[3]]; }
  function sphereSigma(q)   { return Math.acos(Math.min(1.0, Math.abs(q[0]))); }
  const SQ2 = 1 / Math.SQRT2;
  function toQuat(p)        { return [SQ2, p[0]*SQ2, p[1]*SQ2, p[2]*SQ2]; }
  function qrot(q, v) {
    const r = hamilton(hamilton(q, [0,v[0],v[1],v[2]]), sphereInverse(q));
    return [r[1], r[2], r[3]];
  }
  function rotQuat(ax, ay, az, angle) {
    const s = Math.sin(angle/2), l = Math.sqrt(ax*ax+ay*ay+az*az)||1;
    return [Math.cos(angle/2), ax/l*s, ay/l*s, az/l*s];
  }
  function geodesicForce(pos, home, k) {
    const d = pos[0]*home[0]+pos[1]*home[1]+pos[2]*home[2];
    const sigma = Math.acos(Math.max(-1, Math.min(1, d)));
    if (sigma < 0.0001) return [0,0,0];
    const tx = home[0]-d*pos[0], ty = home[1]-d*pos[1], tz = home[2]-d*pos[2];
    const tl = Math.sqrt(tx*tx+ty*ty+tz*tz);
    if (tl < 0.0001) return [0,0,0];
    const s = sigma*k/tl;
    return [tx*s, ty*s, tz*s];
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Panel 1 — DATA INTEGRITY (Closure CLI)
  // Lat/lon grid = organized data, not noise. A scan pulse sweeps each latitude
  // row in sequence — the integrity check traveling through the structure.
  // Sequential glow is exactly what verification looks like live.
  // ─────────────────────────────────────────────────────────────────────────────
  (function initVerification() {
    const canvas = document.getElementById('panel-verify');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W = 0, H = 0, R = 0, lastW = 0, lastH = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 10) return;
      const nw = Math.round(rect.width), nh = Math.round(rect.height);
      if (nw === lastW && nh === lastH) return;
      lastW = W = canvas.width  = nw;
      lastH = H = canvas.height = nh;
      R = Math.min(W, H) * 0.38;
    }
    window.addEventListener('resize', resize);

    let qOrbit   = [1,0,0,0];
    const dqOrbit = rotQuat(0.08, 1.0, 0.04, 0.0010);

    function latLonGrid(rows, cols) {
      const pts = [];
      for (let r = 0; r <= rows; r++) {
        const theta = (r / rows) * Math.PI;
        const y = Math.cos(theta), s = Math.sin(theta);
        const n = (r === 0 || r === rows) ? 1 : cols;
        for (let c = 0; c < n; c++) {
          const phi = (c / n) * Math.PI * 2;
          pts.push([s*Math.cos(phi), y, s*Math.sin(phi)]);
        }
      }
      return pts;
    }

    const grid = latLonGrid(13, 22);
    const pts = grid.map(([x,y,z]) => ({
      home:[x,y,z], pos:[x,y,z], vel:[0,0,0],
      homeQ:toQuat([x,y,z]), sigma:0, scanGlow:0,
      friction:0.93, spring:0.055+Math.random()*0.012,
    }));
    const N = pts.length;

    const edges = [];
    const thr = 0.962;
    for (let i = 0; i < N; i++)
      for (let j = i+1; j < N; j++) {
        const [ax,ay,az]=pts[i].home,[bx,by,bz]=pts[j].home;
        if (ax*bx+ay*by+az*bz > thr) edges.push([i,j]);
      }

    let scanY = 3, scanning = false, nextScan = 2000, lastNow = performance.now();

    function render(now) {
      resize(); if (!W) { requestAnimationFrame(render); return; }
      const dt = now - lastNow; lastNow = now;
      if (!scanning) { nextScan -= dt; if (nextScan<=0) { scanning=true; scanY=-1.7; nextScan=7000+Math.random()*4000; } }
      else { scanY+=0.014; if (scanY>1.7) scanning=false; }

      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,W,H);
      qOrbit = qnorm(hamilton(qOrbit, dqOrbit));

      pts.forEach(p => {
        const [fx,fy,fz]=geodesicForce(p.pos,p.home,p.spring);
        p.vel[0]+=fx; p.vel[1]+=fy; p.vel[2]+=fz;
        p.vel[0]*=p.friction; p.vel[1]*=p.friction; p.vel[2]*=p.friction;
        p.pos[0]+=p.vel[0]*0.08; p.pos[1]+=p.vel[1]*0.08; p.pos[2]+=p.vel[2]*0.08;
        const l=Math.sqrt(p.pos[0]**2+p.pos[1]**2+p.pos[2]**2);
        if(l>0.001){p.pos[0]/=l;p.pos[1]/=l;p.pos[2]/=l;}
        p.sigma=sphereSigma(hamilton(sphereInverse(p.homeQ),toQuat(p.pos)));
        p.scanGlow *= 0.94;
        if (scanning && Math.abs(p.pos[1]-scanY)<0.11) p.scanGlow=Math.max(p.scanGlow,1.0);
      });

      let C=[1,0,0,0]; pts.forEach(p=>{C=compose(C,toQuat(p.pos));}); const gSigma=sphereSigma(C);

      const proj = pts.map((p,i) => {
        const rp=qrot(qOrbit,p.pos), per=1+rp[2]*0.28;
        return {i, x:W/2+rp[0]*R*per, y:H/2-rp[1]*R*per, z:rp[2],
                depth:(rp[2]+1)/2, sigma:p.sigma, sg:p.scanGlow};
      });

      ctx.lineWidth = 0.4;
      edges.forEach(([a,b]) => {
        const pa=proj[a], pb=proj[b];
        if (pa.z < -0.08 && pb.z < -0.08) return;
        const dep=(pa.depth+pb.depth)*0.5;
        const sg=Math.max(pa.sg,pb.sg);
        const hue = sg>0.3 ? 175 : 195;
        const alpha = sg>0.1 ? 0.05+dep*0.18+sg*0.28 : 0.03+dep*0.10;
        ctx.strokeStyle=`hsla(${hue},65%,55%,${alpha})`;
        ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
      });

      proj.sort((a,b)=>a.z-b.z);
      proj.forEach(pp => {
        const sg=pp.sg;
        const hue=sg>0.25?175-sg*20:198-pp.depth*16;
        const sat=55+sg*20+pp.depth*8;
        const lum=30+sg*34+pp.depth*18;
        const alp=0.16+0.76*pp.depth+sg*0.44;
        const sz=Math.max(1.1,1.1+pp.depth*1.4+sg*1.8);
        if (sg>0.15) { ctx.shadowColor=`hsl(${hue},${sat+12}%,${lum+28}%)`; ctx.shadowBlur=Math.min(16,sz*4.5*sg); }
        else ctx.shadowBlur=0;
        ctx.globalAlpha=alp; ctx.fillStyle=`hsl(${hue},${sat}%,${lum}%)`;
        ctx.beginPath(); ctx.arc(pp.x,pp.y,sz,0,Math.PI*2); ctx.fill();
      });
      ctx.shadowBlur=0; ctx.globalAlpha=1;

      const fs=Math.max(10,W*0.027); ctx.font=`${fs}px 'Courier New',monospace`;
      ctx.textAlign='right'; ctx.fillStyle=`rgba(80,195,215,0.60)`;
      ctx.fillText(`\u03c3 = ${gSigma.toFixed(3)}`,W-14,H-14); ctx.textAlign='left';
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  })();

  // ─────────────────────────────────────────────────────────────────────────────
  // Panel 2 — CLOSURE DNA (Storage)
  // True 3-D cylindrical double helix — NOT on S².
  // qrot(qOrbit, pt) is the closure SDK rotating and projecting each node.
  // For σ, each Euclidean point is normalised onto S² before quaternion embed.
  // A scan wave travels the helix index: data being read base by base.
  // ─────────────────────────────────────────────────────────────────────────────
  (function initDNA() {
    const canvas = document.getElementById('panel-storage');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W = 0, H = 0, R = 0, lastW = 0, lastH = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 10) return;
      const nw = Math.round(rect.width), nh = Math.round(rect.height);
      if (nw === lastW && nh === lastH) return;
      lastW = W = canvas.width  = nw;
      lastH = H = canvas.height = nh;
      R = Math.min(W, H) * 0.44;
    }
    window.addEventListener('resize', resize);

    let qOrbit = [1,0,0,0];
    // Rotate around helix axis (Y), slight tilt so depth reads as 3-D
    const dqOrbit = rotQuat(0.06, 1.0, 0.05, 0.0013);

    const N_PT  = 50;
    const TURNS = 2.6;
    const HRAD  = 0.40; // cylinder radius, fraction of R

    // Pure Euclidean cylindrical helix — NOT projected onto S²
    function makeStrand(n, turns, phiOffset) {
      return Array.from({length: n}, (_, i) => {
        const t   = i / (n - 1);
        const phi = t * Math.PI * 2 * turns + phiOffset;
        return [HRAD * Math.cos(phi), t * 2 - 1, HRAD * Math.sin(phi)];
      });
    }

    // Normalise a 3-D vector onto S² (for σ algebra only)
    function vnorm(v) {
      const l = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) || 1;
      return [v[0]/l, v[1]/l, v[2]/l];
    }

    const homes1 = makeStrand(N_PT, TURNS, 0);
    const homes2 = makeStrand(N_PT, TURNS, Math.PI); // anti-parallel strand

    function makePts(homes, strand) {
      return homes.map((home, idx) => ({
        home, pos: [...home], vel: [0, 0, 0],
        homeQ: toQuat(vnorm(home)), // normalised copy lives on S²
        sigma: 0, scanGlow: 0, strand, idx,
        friction: 0.93 + Math.random()*0.03,
        spring:   0.048 + Math.random()*0.012,
      }));
    }

    const pts1 = makePts(homes1, 0);
    const pts2 = makePts(homes2, 1);
    const allPts = [...pts1, ...pts2];

    let scanT = -0.08, scanActive = false, nextScan = 2200;
    let lastNow = performance.now();

    function render(now) {
      resize(); if (!W) { requestAnimationFrame(render); return; }
      const dt = now - lastNow; lastNow = now;

      if (!scanActive) {
        nextScan -= dt;
        if (nextScan <= 0) { scanActive = true; scanT = -0.06; nextScan = 6000+Math.random()*5000; }
      } else {
        scanT += 0.006;
        if (scanT > 1.07) scanActive = false;
      }

      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,W,H);
      qOrbit = qnorm(hamilton(qOrbit, dqOrbit));

      allPts.forEach(p => {
        // Euclidean spring (not geodesic — these points are off-sphere)
        p.vel[0] += (p.home[0] - p.pos[0]) * p.spring;
        p.vel[1] += (p.home[1] - p.pos[1]) * p.spring;
        p.vel[2] += (p.home[2] - p.pos[2]) * p.spring;
        // Micro-wander — keeps the helix organic
        p.vel[0] += (Math.random()-0.5)*0.00070;
        p.vel[1] += (Math.random()-0.5)*0.00070;
        p.vel[2] += (Math.random()-0.5)*0.00070;
        p.vel[0]*=p.friction; p.vel[1]*=p.friction; p.vel[2]*=p.friction;
        p.pos[0]+=p.vel[0]; p.pos[1]+=p.vel[1]; p.pos[2]+=p.vel[2];
        // σ: normalise onto S², then embed as quaternion and compose
        p.sigma = sphereSigma(hamilton(sphereInverse(p.homeQ), toQuat(vnorm(p.pos))));
        p.scanGlow *= 0.90;
        const t = p.idx / (N_PT - 1);
        if (scanActive && Math.abs(t - scanT) < 0.072) p.scanGlow = Math.max(p.scanGlow, 1.0);
      });

      // Global σ from running product of all normalised positions
      let C=[1,0,0,0];
      allPts.forEach(p => { C = compose(C, toQuat(vnorm(p.pos))); });
      const gSigma = sphereSigma(C);

      // qrot IS the closure SDK step: rotate each Euclidean point, then project
      function project(pts) {
        return pts.map(p => {
          const rp  = qrot(qOrbit, p.pos);   // SDK rotation
          const per = 1 + rp[2] * 0.55;      // stronger perspective than globe
          return { x: W/2+rp[0]*R*per, y: H/2-rp[1]*R*per,
                   z: rp[2], depth: (rp[2]+1)/2,
                   sg: p.scanGlow, strand: p.strand, idx: p.idx };
        });
      }
      const proj1 = project(pts1);
      const proj2 = project(pts2);

      // ── Rungs every 4 base-pairs ───────────────────────────────────────────
      for (let i = 0; i < N_PT; i += 4) {
        const pa = proj1[i], pb = proj2[i];
        if (!pa || !pb) continue;
        const dep  = (pa.depth + pb.depth) * 0.5;
        const sg   = Math.max(pa.sg, pb.sg);
        const fade = 0.18 + dep * 0.72; // back-facing rungs go nearly invisible
        ctx.lineWidth = 0.65 + sg*1.6;
        ctx.strokeStyle = `hsla(200,44%,60%,${(0.06+dep*0.22+sg*0.42)*fade})`;
        ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
      }

      // ── Strand backbones ───────────────────────────────────────────────────
      function drawBackbone(proj, hue) {
        ctx.lineWidth = 1.2;
        for (let i = 0; i < proj.length-1; i++) {
          const pa=proj[i], pb=proj[i+1];
          const dep=(pa.depth+pb.depth)*0.5;
          const sg=Math.max(pa.sg,pb.sg);
          const fade = 0.18 + dep * 0.82; // back half nearly invisible
          ctx.strokeStyle=`hsla(${hue},56%,46%,${(0.10+dep*0.52+sg*0.26)*fade})`;
          ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
        }
      }
      drawBackbone(proj1, 162); // teal
      drawBackbone(proj2, 252); // indigo

      // ── Nodes depth-sorted ─────────────────────────────────────────────────
      const allProj = [...proj1, ...proj2].sort((a,b) => a.z - b.z);
      allProj.forEach(pp => {
        const hue  = pp.strand === 0 ? 158 : 256;
        const sg = pp.sg, dep = pp.depth;
        const fade = 0.15 + dep * 0.85;
        const sat  = 55 + sg*18 + dep*7;
        const lum  = 36 + dep*22 + sg*22;
        const alp  = (0.16 + 0.76*dep + sg*0.42) * fade;
        const sz   = Math.max(1.2, 1.4+dep*1.8+sg*2.4);
        if (sg > 0.10) {
          ctx.shadowColor = `hsl(${hue},${sat}%,${lum+30}%)`;
          ctx.shadowBlur  = Math.min(22, sz*5.0*sg);
        } else ctx.shadowBlur = 0;
        ctx.globalAlpha = alp;
        ctx.fillStyle   = `hsl(${hue},${sat}%,${lum}%)`;
        ctx.beginPath(); ctx.arc(pp.x, pp.y, sz, 0, Math.PI*2); ctx.fill();
      });

      ctx.shadowBlur=0; ctx.globalAlpha=1;
      const fs=Math.max(10,W*0.027); ctx.font=`${fs}px 'Courier New',monospace`;
      ctx.textAlign='right';
      ctx.fillStyle=`rgba(62,185,162,0.62)`;
      ctx.fillText(`\u03c3 = ${gSigma.toFixed(3)}`,W-14,H-14); ctx.textAlign='left';
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  })();

  // ─────────────────────────────────────────────────────────────────────────────
  // Panel 3 — LEARNING (Enkidu / Torus)
  // The state space of hunger × cold IS a torus. pos2quat(x,y) in enkidu_alive
  // embeds it as the Clifford torus in S³:
  //   q = [cos(a)cos(b), sin(a)cos(b), cos(a)sin(b), sin(a)sin(b)]
  // σ = geodesic distance from identity = free energy = how far from homeostasis.
  // The particle is Enkidu's state drifting outward as drives accumulate,
  // snapping back when they resolve. The torus refuses to close — until it does.
  // ─────────────────────────────────────────────────────────────────────────────
  (function initTorus() {
    const canvas = document.getElementById('panel-learn');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W = 0, H = 0, R = 0, lastW = 0, lastH = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 10) return;
      const nw = Math.round(rect.width), nh = Math.round(rect.height);
      if (nw === lastW && nh === lastH) return;
      lastW = W = canvas.width  = nw;
      lastH = H = canvas.height = nh;
      R = Math.min(W, H) * 0.40;
    }
    window.addEventListener('resize', resize);

    // Pre-tilt ~62° around X so the donut faces the viewer horizontally
    let qOrbit = rotQuat(1, 0, 0, 1.08);
    // Spin mostly around Z (screen vertical) — torus keeps its horizontal pose
    const dqOrbit = rotQuat(0.08, 0.18, 1.0, 0.0016);

    // Torus geometry: standard donut in Euclidean 3-D
    const TMAJ = 0.58, TMIN = 0.24;
    const U_SEGS = 20, V_SEGS = 11;

    function torusPt(u, v) {
      return [
        (TMAJ + TMIN*Math.cos(v)) * Math.cos(u),
        (TMAJ + TMIN*Math.cos(v)) * Math.sin(u),
        TMIN * Math.sin(v),
      ];
    }

    // Clifford torus σ: the free energy of state (hunger=u, cold=v)
    function cliffordSigma(u, v) {
      return Math.acos(Math.min(1.0, Math.abs(Math.cos(u) * Math.cos(v))));
    }

    // Precompute mesh grid with σ at each vertex
    const mesh = [];
    for (let ui = 0; ui < U_SEGS; ui++) {
      for (let vi = 0; vi < V_SEGS; vi++) {
        const u = (ui / U_SEGS) * Math.PI * 2;
        const v = (vi / V_SEGS) * Math.PI * 2;
        mesh.push({ pt: torusPt(u, v), sig: cliffordSigma(u, v) });
      }
    }

    // Two angles winding continuously mod 2π — ratio ≈ golden ratio φ so the
    // path is quasi-periodic and covers the torus surface without ever repeating.
    const D_HUNGER = 0.0050;
    const D_COLD   = 0.0031; // D_HUNGER / φ  ≈ 0.0031

    let hunger = 0, cold = 0;

    const TRAIL_LEN = 90; // longer trail shows the winding spiral clearly
    const trail = [];

    let lastNow = performance.now();

    function render(now) {
      resize(); if (!W) { requestAnimationFrame(render); return; }
      const dt = now - lastNow; lastNow = now;
      const s = dt / 16; // normalise to 60fps

      // Smooth continuous winding — no jumps, no resets
      hunger = (hunger + D_HUNGER * s) % (Math.PI * 2);
      cold   = (cold   + D_COLD   * s) % (Math.PI * 2);

      // Clifford torus embedding — exact algebra from enkidu_alive_sdk
      const enkQ = [
        Math.cos(cold)*Math.cos(hunger),
        Math.sin(cold)*Math.cos(hunger),
        Math.cos(cold)*Math.sin(hunger),
        Math.sin(cold)*Math.sin(hunger),
      ];
      const enkSigma = sphereSigma(enkQ);

      // Visual position directly on the torus surface
      const enkPt = torusPt(hunger, cold);
      trail.push([...enkPt]);
      if (trail.length > TRAIL_LEN) trail.shift();

      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,W,H);
      qOrbit = qnorm(hamilton(qOrbit, dqOrbit));

      function proj(pt) {
        const rp = qrot(qOrbit, pt);
        const per = 1 + rp[2] * 0.50;
        return { x: W/2+rp[0]*R*per, y: H/2-rp[1]*R*per, z: rp[2], depth: (rp[2]+1)/2 };
      }

      // ── Torus mesh: thin solid edges + depth-sorted dots (same pattern as panels 1 & 2)
      ctx.lineWidth = 0.45;

      // v-direction edges (longitude circles)
      for (let ui = 0; ui < U_SEGS; ui++) {
        const nu = (ui+1) % U_SEGS;
        for (let vi = 0; vi < V_SEGS; vi++) {
          const pa = proj(mesh[ui*V_SEGS+vi].pt);
          const pb = proj(mesh[nu*V_SEGS+vi].pt);
          if (pa.z < -0.85 && pb.z < -0.85) continue;
          const dep = (pa.depth+pb.depth)*0.5;
          const sg  = (mesh[ui*V_SEGS+vi].sig + mesh[nu*V_SEGS+vi].sig) * 0.5;
          const t   = sg / (Math.PI*0.5);
          const hue = 198 - t*20;
          ctx.strokeStyle = `hsla(${hue},60%,55%,${0.04+dep*0.12})`;
          ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
        }
      }
      // u-direction edges (tube circles)
      for (let ui = 0; ui < U_SEGS; ui++) {
        for (let vi = 0; vi < V_SEGS; vi++) {
          const nv = (vi+1) % V_SEGS;
          const pa = proj(mesh[ui*V_SEGS+vi].pt);
          const pb = proj(mesh[ui*V_SEGS+nv].pt);
          if (pa.z < -0.85 && pb.z < -0.85) continue;
          const dep = (pa.depth+pb.depth)*0.5;
          ctx.strokeStyle = `hsla(195,55%,52%,${0.03+dep*0.10})`;
          ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
        }
      }

      // Depth-sorted vertex dots — the main visual, same as panels 1 & 2
      const verts = mesh.map((m) => {
        const p = proj(m.pt);
        return { x: p.x, y: p.y, z: p.z, depth: p.depth, sig: m.sig };
      }).sort((a,b) => a.z - b.z);

      verts.forEach(vp => {
        const t   = vp.sig / (Math.PI*0.5);
        const hue = 198 - t*20;
        const dep = vp.depth;
        const sat = 58 + dep*8;
        const lum = 42 + dep*18;
        const alp = 0.14 + dep*0.72;
        const sz  = Math.max(1.1, 1.1 + dep*1.5);
        ctx.shadowBlur = 0;
        ctx.globalAlpha = alp;
        ctx.fillStyle = `hsl(${hue},${sat}%,${lum}%)`;
        ctx.beginPath(); ctx.arc(vp.x, vp.y, sz, 0, Math.PI*2); ctx.fill();
      });
      ctx.globalAlpha = 1;

      // ── Trail — shows the quasi-periodic winding on the torus ─────────────
      ctx.lineWidth = 1.0;
      for (let i = 1; i < trail.length; i++) {
        const pa = proj(trail[i-1]), pb = proj(trail[i]);
        const t  = i / trail.length;
        ctx.strokeStyle = `hsla(195,52%,52%,${t*0.45})`;
        ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
      }

      // ── Enkidu state-point ─────────────────────────────────────────────────
      const ep  = proj(enkPt);
      const t2  = enkSigma / (Math.PI*0.5);
      const eHue = 195 - t2*20; // teal, consistent with panel 1 & 2
      ctx.shadowColor = `hsl(${eHue},62%,68%)`;
      ctx.shadowBlur  = 8 + t2*6;
      ctx.globalAlpha = 0.90;
      ctx.fillStyle   = `hsl(${eHue},58%,58%)`;
      ctx.beginPath(); ctx.arc(ep.x, ep.y, 3.2, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur=0; ctx.globalAlpha=1;
      const fs=Math.max(10,W*0.027); ctx.font=`${fs}px 'Courier New',monospace`;
      ctx.textAlign='right';
      ctx.fillStyle=`hsla(${210-t2*165},65%,58%,0.65)`;
      ctx.fillText(`\u03c3 = ${enkSigma.toFixed(3)}`,W-14,H-14); ctx.textAlign='left';
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  })();

})();
