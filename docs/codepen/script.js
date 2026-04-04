const canvas = document.getElementById("webgl-canvas");
const gl = canvas.getContext("webgl", { alpha: false });

if (!gl) {
  canvas.style.background = "#0a0a0f";
  throw new Error("WebGL not supported");
}

const vs = `attribute vec2 a; void main(){ gl_Position=vec4(a,0,1); }`;

const fs = `
precision highp float;
uniform vec2 uR;
uniform float uT, uS, uSc, uBl;
uniform vec3 uBg;

#define TAU 6.2831853

mat2 r2(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, -s, s, c);
}

float sphere(vec3 p, float r) { return length(p) - r; }

float torus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

float box(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.);
}

float octa(vec3 p, float s) {
  p = abs(p);
  return (p.x + p.y + p.z - s) * .5773;
}

float sdf(vec3 p) {
  float t = uT * .25, sc = uSc, bl = uBl;

  float d0 = sphere(p, .65 + .05 * sin(t * 1.3));

  vec3 p1 = p; p1.xz = r2(t * .6) * p1.xz;
  float d1 = torus(p1, vec2(.55, .22));

  vec3 p2 = p; p2.xy = r2(t * .4) * p2.xy; p2.yz = r2(t * .3) * p2.yz;
  float d2 = box(p2, vec3(.42 + .04 * sin(t * 2.)));

  vec3 p3 = p; p3.xy = r2(t * .5) * p3.xy;
  float d3 = octa(p3, .72 + .04 * sin(t * 1.7));

  vec3 p4 = p; p4.xz = r2(t * .7) * p4.xz;
  float d4a = torus(p4, vec2(.45, .15));
  vec3 p5 = p; p5.xy = r2(t * .5 + 1.2) * p5.xy;
  float d4b = torus(p5, vec2(.35, .12));
  float d4 = min(d4a, d4b);

  if (sc < 1.) return mix(d0, d1, bl);
  if (sc < 2.) return mix(d1, d2, bl);
  if (sc < 3.) return mix(d2, d3, bl);
  return mix(d3, d4, bl);
}

vec3 norm(vec3 p) {
  float e = .001;
  return normalize(vec3(
    sdf(p + vec3(e,0,0)) - sdf(p - vec3(e,0,0)),
    sdf(p + vec3(0,e,0)) - sdf(p - vec3(0,e,0)),
    sdf(p + vec3(0,0,e)) - sdf(p - vec3(0,0,e))
  ));
}

vec3 pal(float t) {
  return .5 + .5 * cos(TAU * (.9 * t + vec3(0., .15, .25)));
}

void main() {
  vec2 uv = (gl_FragCoord.xy - uR * .5) / min(uR.x, uR.y);
  vec3 ro = vec3(0, 0, 2.4);
  vec3 rd = normalize(vec3(uv, -1.2));

  float t = 0., hit = 0.;
  for (int i = 0; i < 96; i++) {
    float d = sdf(ro + rd * t);
    if (d < .001) { hit = 1.; break; }
    if (t > 6.) break;
    t += d;
  }

  vec3 bg = uBg, col = bg;

  if (hit > .5) {
    vec3 p = ro + rd * t;
    vec3 n = norm(p);
    vec3 bc = pal(uS);
    vec3 l = normalize(vec3(.7, 1., .5));
    float dif = clamp(dot(n, l), 0., 1.);
    float spe = pow(clamp(dot(reflect(-l, n), -rd), 0., 1.), 32.);
    float fr  = pow(1. - clamp(dot(-rd, n), 0., 1.), 3.5);
    col = bc * (dif * .7 + .3) + spe * .5 + fr * vec3(.784, 1., .278) * .6;
    col = mix(bg, col, exp(-t * .15));
  }

  col = mix(uBg, col, clamp(1. - dot(uv * .9, uv * .9), 0., 1.));
  col += (fract(sin(dot(gl_FragCoord.xy, vec2(127.1, 311.7))) * 43758.5) - .5) * .025;

  gl_FragColor = vec4(col, 1.);
}`;

const mkShader = (type, src) => {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
};

const prog = gl.createProgram();
gl.attachShader(prog, mkShader(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, mkShader(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  console.error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);

gl.uniform3f(gl.getUniformLocation(prog, "uBg"), 0.039, 0.039, 0.059);

const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(
  gl.ARRAY_BUFFER,
  new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
  gl.STATIC_DRAW
);
const ap = gl.getAttribLocation(prog, "a");
gl.enableVertexAttribArray(ap);
gl.vertexAttribPointer(ap, 2, gl.FLOAT, false, 0, 0);

const uR = gl.getUniformLocation(prog, "uR");
const uTi = gl.getUniformLocation(prog, "uT");
const uScroll = gl.getUniformLocation(prog, "uS");
const uScene = gl.getUniformLocation(prog, "uSc");
const uBlend = gl.getUniformLocation(prog, "uBl");
const uBg = gl.getUniformLocation(prog, "uBg");

let maxScroll = 1;

const resize = () => {
  const dpr = Math.min(devicePixelRatio ?? 1, 2);
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  canvas.style.width = `${innerWidth}px`;
  canvas.style.height = `${innerHeight}px`;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.uniform2f(uR, canvas.width, canvas.height);
  maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight);
};
resize();
window.addEventListener("resize", resize);

const N = 5;
const NAMES = ["SCENE 01", "SCENE 02", "SCENE 03", "SCENE 04", "SCENE 05"];

let tgt = 0;
let smooth = 0;
let velocity = 0;

const ease = 0.1;
const mq = window.matchMedia("(prefers-color-scheme: dark)");

window.addEventListener(
  "scroll",
  () => {
    tgt = maxScroll > 0 ? scrollY / maxScroll : 0;
  },
  { passive: true }
);

window.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const linePx = 16;
    const pagePx = innerHeight * 0.9;
    const delta =
      e.deltaMode === 1
        ? e.deltaY * linePx
        : e.deltaMode === 2
        ? e.deltaY * pagePx
        : e.deltaY;
    velocity += delta;
    velocity = Math.max(-600, Math.min(600, velocity));
  },
  { passive: false }
);

const progFill = document.getElementById("prog-fill");
const hudPct = document.getElementById("hud-pct");
const sceneName = document.getElementById("scene-name");
const dots = document.querySelectorAll(".scene-dot");

const updateHUD = (s) => {
  const p = Math.round(s * 100);
  hudPct.textContent = String(p).padStart(3, "0") + "%";
  progFill.style.width = `${p}%`;
  const si = Math.min(N - 1, Math.floor(s * N));
  sceneName.textContent = NAMES[si];
  dots.forEach((d, i) => d.classList.toggle("active", i === si));
};

const revealEls = [
  ...document.querySelectorAll(
    ".tag, h1, h2, .body-text, .stat-row, .cta, .h-line"
  )
];

revealEls.forEach((el) => {
  if (el.getBoundingClientRect().top < innerHeight * 0.92)
    el.classList.add("visible");
});

const io = new IntersectionObserver(
  (entries) =>
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        io.unobserve(e.target);
      }
    }),
  { threshold: 0.1 }
);
revealEls.forEach((el) => io.observe(el));

const hexToVec3 = (hex) => {
  const n = parseInt(hex.replace("#", ""), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

const bgColors = { dark: "#0a0a0f", light: "#f0ece3" };

const updateBg = (theme) => {
  const [r, g, b] = hexToVec3(bgColors[theme] ?? bgColors.dark);
  gl.uniform3f(uBg, r, g, b);
};

const getSystemTheme = () => (mq.matches ? "dark" : "light");

const applyTheme = (theme) => {
  document.documentElement.setAttribute("data-theme", theme);
  document.documentElement.style.colorScheme = theme;
  updateBg(theme);
};

applyTheme(getSystemTheme());

mq.addEventListener("change", (e) => {
  applyTheme(e.matches ? "dark" : "light");
});

document.getElementById("theme-toggle").addEventListener("click", () => {
  const current =
    document.documentElement.getAttribute("data-theme") || getSystemTheme();
  applyTheme(current === "dark" ? "light" : "dark");
});

const t0 = performance.now();
let lastNow = t0;

const frame = (now) => {
  requestAnimationFrame(frame);

  const dt = Math.min((now - lastNow) / 1000, 0.05);
  lastNow = now;

  velocity *= Math.pow(0.85, dt * 60);
  if (Math.abs(velocity) > 0.2)
    window.scrollBy({ top: velocity * ease, behavior: "auto" });

  smooth += (tgt - smooth) * (1 - Math.exp(-dt * 8));

  const raw = smooth * (N - 1);
  const flr = Math.floor(raw);
  const si = Math.min(flr, N - 2);
  const bl = flr >= N - 1 ? 1.0 : raw - flr;

  updateHUD(smooth);

  gl.uniform1f(uTi, (now - t0) / 1000);
  gl.uniform1f(uScroll, smooth);
  gl.uniform1f(uScene, si);
  gl.uniform1f(uBlend, bl);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
};

requestAnimationFrame(frame);

let anchorAnim = null;

const stopAnchorAnim = () => {
  if (anchorAnim) {
    cancelAnimationFrame(anchorAnim);
    anchorAnim = null;
  }
};

const smoothScrollToY = (targetY, duration = 900) => {
  stopAnchorAnim();
  velocity = 0;

  const startY = window.scrollY;
  const diff = targetY - startY;
  const start = performance.now();

  const easeInOutCubic = (t) =>
    t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const e = easeInOutCubic(p);

    window.scrollTo(0, startY + diff * e);

    if (p < 1) {
      anchorAnim = requestAnimationFrame(tick);
    } else {
      anchorAnim = null;
    }
  };

  anchorAnim = requestAnimationFrame(tick);
};

window.addEventListener("wheel", stopAnchorAnim, { passive: true });
window.addEventListener("touchstart", stopAnchorAnim, { passive: true });
window.addEventListener("mousedown", stopAnchorAnim, { passive: true });
window.addEventListener("keydown", stopAnchorAnim);

document.querySelectorAll('a[href^="#s"]').forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();

    const id = a.getAttribute("href");
    const target = document.querySelector(id);
    if (!target) return;

    const y = Math.max(0, Math.min(target.offsetTop, maxScroll));
    smoothScrollToY(y);
  });
});
