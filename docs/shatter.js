// Particulate shatter — ORI ring only, no UI chrome
// hover = magnet pull, click/hold = blow, releases back to ring

(function () {
  const canvas = document.getElementById("ori-canvas");
  const ctx    = canvas.getContext("2d", { willReadFrequently: true });
  let W, H;

  let particles    = [];
  let mx = 0, my = 0;
  let isDown       = false;

  // ── Resize canvas to fill its container ──────────────────────────────────
  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 480);
    W = canvas.width  = size;
    H = canvas.height = size;
    canvas.style.width  = size + "px";
    canvas.style.height = size + "px";
  }

  window.addEventListener("resize", () => {
    resize();
    buildParticles();
  });

  // ── Particle ──────────────────────────────────────────────────────────────
  class Particle {
    constructor(ox, oy, r, g, b, size) {
      this.x = ox; this.y = oy;
      this.ox = ox; this.oy = oy;
      this.r = r; this.g = g; this.b = b;
      this.size = size; this.baseSize = size;

      this.vx = (Math.random() - 0.5) * 0.07;
      this.vy = (Math.random() - 0.5) * 0.07;

      this.friction      = 0.94 + Math.random() * 0.03;
      this.spring        = 0.007 + Math.random() * 0.003;
      this.wanderAngle   = Math.random() * Math.PI * 2;
      this.wanderSpeed   = 0.005 + Math.random() * 0.005;
      this.opacity       = 1;
    }

    update() {
      this.opacity = 1;

      // spring toward origin
      this.vx += (this.ox - this.x) * this.spring;
      this.vy += (this.oy - this.y) * this.spring;

      // gentle wander
      this.wanderAngle += this.wanderSpeed;
      this.vx += Math.cos(this.wanderAngle) * 0.007;
      this.vy += Math.sin(this.wanderAngle) * 0.007;

      // mouse interaction — click/hold only
      if (isDown) {
        const dx   = this.x - mx;
        const dy   = this.y - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const radius = 50;
        if (dist < radius && dist > 0) {
          const force = (radius - dist) / radius;
          const angle = Math.atan2(dy, dx);
          this.vx += Math.cos(angle) * force * force * 1.4;
          this.vy += Math.sin(angle) * force * force * 1.4;
          this.size = this.baseSize * (1 + force * 0.9);
        } else {
          this.size += (this.baseSize - this.size) * 0.1;
        }
      } else {
        this.size += (this.baseSize - this.size) * 0.1;
      }

      this.vx *= this.friction;
      this.vy *= this.friction;
      this.x  += this.vx;
      this.y  += this.vy;
    }

    draw() {
      ctx.globalAlpha = this.opacity;
      ctx.fillStyle = `rgb(${this.r},${this.g},${this.b})`;
      const s    = Math.max(1, this.size);
      const half = s / 2;
      const rad  = s > 4 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(this.x - half + rad, this.y - half);
      ctx.lineTo(this.x + half - rad, this.y - half);
      ctx.quadraticCurveTo(this.x + half, this.y - half, this.x + half, this.y - half + rad);
      ctx.lineTo(this.x + half, this.y + half - rad);
      ctx.quadraticCurveTo(this.x + half, this.y + half, this.x + half - rad, this.y + half);
      ctx.lineTo(this.x - half + rad, this.y + half);
      ctx.quadraticCurveTo(this.x - half, this.y + half, this.x - half, this.y + half - rad);
      ctx.lineTo(this.x - half, this.y - half + rad);
      ctx.quadraticCurveTo(this.x - half, this.y - half, this.x - half + rad, this.y - half);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  // ── Build particles from the ring image ───────────────────────────────────
  function buildParticles() {
    if (!window._oriImg) return;
    const img = window._oriImg;
    particles  = [];

    const scale = Math.min((W * 0.88) / img.width, (H * 0.88) / img.height, 1);
    const iw    = Math.floor(img.width  * scale);
    const ih    = Math.floor(img.height * scale);
    const ox    = Math.floor((W - iw) / 2);
    const oy    = Math.floor((H - ih) / 2);

    const oc   = document.createElement("canvas");
    oc.width   = iw; oc.height = ih;
    const octx = oc.getContext("2d");
    octx.drawImage(img, 0, 0, iw, ih);
    const data = octx.getImageData(0, 0, iw, ih).data;

    const target = Math.min(6000, Math.max(2000, (iw * ih) / 20));
    const gap    = Math.max(2, Math.floor(Math.sqrt((iw * ih) / target)));
    const pSize  = gap * 0.95;

    for (let y = 0; y < ih; y += gap) {
      for (let x = 0; x < iw; x += gap) {
        const i = (y * iw + x) * 4;
        if (data[i + 3] < 128) continue;       // skip transparent (hole + outside)
        particles.push(new Particle(
          ox + x, oy + y,
          data[i], data[i + 1], data[i + 2],
          pSize
        ));
      }
    }
  }

  // ── Render loop ───────────────────────────────────────────────────────────
  function render() {
    ctx.fillStyle = "rgba(255,255,255,0.28)";
    ctx.fillRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(render);
  }

  // ── Events ────────────────────────────────────────────────────────────────
  canvas.addEventListener("pointermove", e => {
    const r = canvas.getBoundingClientRect();
    mx = e.clientX - r.left;
    my = e.clientY - r.top;
  });
  canvas.addEventListener("pointerleave", () => isDown = false);
  canvas.addEventListener("pointerdown",  () => isDown = true);
  canvas.addEventListener("pointerup",    () => isDown = false);

  // ── Init ──────────────────────────────────────────────────────────────────
  resize();

  const img      = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    window._oriImg = img;
    buildParticles();
    render();
  };
  img.src = "gradient-circle.png";

})();
