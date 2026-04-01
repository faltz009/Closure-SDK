new p5(function (p) {

  // ── Original geometry (unchanged from CodePen) ────────────────────────────
  const SEGMENTS      = 400;
  const RIBBON_HALF_W = 14;
  const X_SCALE       = 1.4;
  const X_OFFSET      = 0.2;

  // ── Original wave / motion (unchanged) ───────────────────────────────────
  const WAVE_SPEED        = 0.018;
  const WAVE1_FREQ        = 3.5;
  const WAVE1_TIME_SPEED  = 0.7;
  const WAVE1_AMP         = 110;
  const WAVE2_FREQ        = 7.0;
  const WAVE2_TIME_SPEED  = 1.1;
  const WAVE2_AMP         = 30;

  // ── Original twist (unchanged) ────────────────────────────────────────────
  const TWIST_CYCLES      = 6;
  const TWIST_TIME_SPEED  = 0.5;
  const FACE_BLEND_GAMMA  = 1.2;

  // ── Original shadow / edges (unchanged) ──────────────────────────────────
  const SHADOW_ALPHA   = 14;
  const SHADOW_DX      = 4;
  const SHADOW_DY      = 7;
  const EDGE_MIN_TWIST = 0.08;
  const EDGE_ALPHA     = 22;
  const EDGE_WEIGHT    = 0.5;

  // ── ORI ring colors (only change from original) ───────────────────────────
  // blue(210°) → purple → red → yellow → green — 300° arc, slow drift
  const HUE_START = 210;
  const HUE_SWEEP = 300;
  const HUE_DRIFT = 22;

  let t = 0;
  let W, H;

  // ── Spine ─────────────────────────────────────────────────────────────────
  function buildSpine(time) {
    const pts = [];
    for (let i = 0; i <= SEGMENTS; i++) {
      const u = i / SEGMENTS;
      pts.push({
        x: u * W * X_SCALE - W * X_OFFSET,
        y: H / 2
          + Math.sin(u * Math.PI * WAVE1_FREQ + time * WAVE1_TIME_SPEED) * WAVE1_AMP
          + Math.sin(u * Math.PI * WAVE2_FREQ + time * WAVE2_TIME_SPEED) * WAVE2_AMP,
      });
    }
    return pts;
  }

  // ── Normals ───────────────────────────────────────────────────────────────
  function buildNormals(pts) {
    const last = pts.length - 1;
    return pts.map((_, i) => {
      const dx = i === 0    ? pts[1].x - pts[0].x
               : i === last ? pts[last].x - pts[last - 1].x
               :              pts[i + 1].x - pts[i - 1].x;
      const dy = i === 0    ? pts[1].y - pts[0].y
               : i === last ? pts[last].y - pts[last - 1].y
               :              pts[i + 1].y - pts[i - 1].y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      return { nx: -dy / len, ny: dx / len };
    });
  }

  // ── Edges with twist (original logic) ────────────────────────────────────
  function buildEdges(pts, normals, time) {
    const tops = [], bots = [], twists = [];
    for (let i = 0; i <= SEGMENTS; i++) {
      const twist = Math.cos((i / SEGMENTS) * Math.PI * TWIST_CYCLES + time * TWIST_TIME_SPEED);
      const w     = RIBBON_HALF_W * Math.abs(twist);
      const sign  = twist >= 0 ? 1 : -1;
      twists.push(twist);
      tops.push({ x: pts[i].x + normals[i].nx * w * sign, y: pts[i].y + normals[i].ny * w * sign });
      bots.push({ x: pts[i].x - normals[i].nx * w * sign, y: pts[i].y - normals[i].ny * w * sign });
    }
    return { tops, bots, twists };
  }

  // ── Color: ORI gradient modulated by twist (replaces original palette) ───
  function getRibbonColor(frac, twist, time) {
    const hue       = (HUE_START + frac * HUE_SWEEP + time * HUE_DRIFT) % 360;
    const facedness = Math.pow(Math.abs(twist), FACE_BLEND_GAMMA);
    const sat       = 82 - facedness * 18;
    const bri       = 70 + facedness * 25;
    return [hue, sat, bri];
  }

  function quad(ax, ay, bx, by, cx, cy, dx, dy) {
    p.beginShape();
    p.vertex(ax, ay); p.vertex(bx, by);
    p.vertex(cx, cy); p.vertex(dx, dy);
    p.endShape(p.CLOSE);
  }

  function drawShadow(tops, bots) {
    p.noStroke();
    for (let i = 0; i < SEGMENTS; i++) {
      p.fill(0, 0, 30, SHADOW_ALPHA);
      quad(
        tops[i].x   + SHADOW_DX, tops[i].y   + SHADOW_DY,
        tops[i+1].x + SHADOW_DX, tops[i+1].y + SHADOW_DY,
        bots[i+1].x + SHADOW_DX, bots[i+1].y + SHADOW_DY,
        bots[i].x   + SHADOW_DX, bots[i].y   + SHADOW_DY
      );
    }
  }

  function drawRibbon(tops, bots, twists, time) {
    for (let i = 0; i < SEGMENTS; i++) {
      const [h, s, b] = getRibbonColor(i / SEGMENTS, twists[i], time);
      p.fill(h, s, b);
      p.noStroke();
      quad(
        tops[i].x,   tops[i].y,
        tops[i+1].x, tops[i+1].y,
        bots[i+1].x, bots[i+1].y,
        bots[i].x,   bots[i].y
      );

      if (Math.abs(twists[i]) > EDGE_MIN_TWIST) {
        p.stroke(0, 0, 20, EDGE_ALPHA);
        p.strokeWeight(EDGE_WEIGHT);
        p.line(tops[i].x, tops[i].y, tops[i+1].x, tops[i+1].y);
        p.line(bots[i].x, bots[i].y, bots[i+1].x, bots[i+1].y);
        p.noStroke();
      }
    }
  }

  // ── p5 lifecycle ──────────────────────────────────────────────────────────

  p.setup = function () {
    W = p.windowWidth;
    H = p.windowHeight;
    const cvs = p.createCanvas(W, H);
    cvs.parent("ribbon-bg");
    p.colorMode(p.HSB, 360, 100, 100, 255);
    p.smooth();
  };

  p.windowResized = function () {
    W = p.windowWidth;
    H = p.windowHeight;
    p.resizeCanvas(W, H);
  };

  p.draw = function () {
    p.background(0, 0, 100);
    t += WAVE_SPEED;

    const pts              = buildSpine(t);
    const normals          = buildNormals(pts);
    const { tops, bots, twists } = buildEdges(pts, normals, t);

    drawShadow(tops, bots);
    drawRibbon(tops, bots, twists, t);
  };

}, "ribbon-bg");
