const ASSET_VERSION = document.querySelector('meta[name="closure-asset-version"]')?.content || 'dev';
const versioned = (url) => `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(ASSET_VERSION)}`;

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.getElementById('seeing-code-canvas');
const metricStep = document.getElementById('metric-step');
const metricSigma = document.getElementById('metric-sigma');
const metricBranch = document.getElementById('metric-branch');
const metricProjection = document.getElementById('metric-product');
const pieceTitle = document.getElementById('piece-title');
const pieceDescription = document.getElementById('piece-description');
const pieceCaption = document.getElementById('piece-caption');
const pieceSource = document.getElementById('piece-source');
const galleryHost = document.getElementById('seeing-gallery');

if (!canvas) {
  throw new Error('seeing-code canvas missing');
}

function stereographicSouth(q, scale = 4.4) {
  const [w, x, y, z] = q;
  const denom = Math.max(1e-6, 1 + w);
  return new THREE.Vector3((x / denom) * scale, (y / denom) * scale, (z / denom) * scale);
}

function catmullPath(points) {
  const padded = [points[0].clone()].concat(points, [points[points.length - 1].clone()]);
  return new THREE.CatmullRomCurve3(padded, false, 'catmullrom', 0.18);
}

const PIECE_META = {
  piece1: {
    title: 'Append Inverse',
    cardBlurb: 'A short program closes into a loop.',
    description:
      'The program is executed in closure_ea/vm. After each instruction, we save the actual machine state on S³. The final appended inverse closes the trace back to identity, so the program becomes a sealed spatial loop.',
    caption:
      'Blue points are ordinary execution states. The amber point is the appended inverse. The green point is closure at identity. The camera drifts so the code reads as an object.',
    source: [
      'q1 = quat(0.74, i)',
      'q2 = quat(0.92, j)',
      'q3 = quat(0.58, k)',
      'C  = q1 * q2 * q3',
      'program = [q1, q2, q3, inverse(C)]',
    ].join('\n'),
  },
  piece2: {
    title: 'Folded Loop',
    cardBlurb: 'A longer program folding back on itself.',
    description:
      'This is a longer real VM program: twelve quaternion instructions across all three axes, then the appended inverse of the compiled product. The trace folds back on itself repeatedly before sealing shut.',
    caption:
      'The second piece is denser because the executed path revisits nearby regions of S³ before the inverse closes the whole body. Same engine, same projection, richer code-shape.',
    source: [
      'program = [',
      '  quat(0.88, i), quat(1.06, j), quat(-0.63, k),',
      '  quat(0.71, j), quat(-0.92, i), quat(0.57, k),',
      '  quat(0.49, i), quat(-0.84, j), quat(1.11, k),',
      '  quat(-0.68, i), quat(0.77, j), quat(-0.53, k),',
      '  inverse(compiled(program))',
      ']',
    ].join('\n'),
  },
  piece3: {
    title: 'Multiply',
    cardBlurb: 'A tiny function written into shape.',
    description:
      'This trace comes from an actual Python multiplication function embedded token by token through closure_rs.path_from_raw_bytes("Sphere", ..., hashed=False). The geometry is the source body itself.',
    caption:
      'These points come from the source tokens of a tiny function. They show how everyday code acquires shape once it is written into the Closure substrate.',
    source: 'def multiply(a, b):\n    result = a * b\n    return result',
  },
  piece4: {
    title: 'Integrate',
    cardBlurb: 'A longer function with a denser trace.',
    description:
      'This piece runs a real integration snippet through the same Rust path builder, again token by token. Because the source is longer and structurally repetitive, the resulting body is denser and more woven.',
    caption:
      'Longer source with loops and arithmetic creates a richer trace. Every point still comes directly from the engine-generated path over the source tokens.',
    source: [
      'def integrate(f, a, b, n=256):',
      '    dx = (b - a) / n',
      '    total = 0.0',
      '    for i in range(n):',
      '        x = a + (i + 0.5) * dx',
      '        total += f(x) * dx',
      '    return total',
    ].join('\n'),
  },
};

function applyPieceText(key) {
  const meta = PIECE_META[key];
  pieceTitle.textContent = meta.title;
  pieceDescription.textContent = meta.description;
  pieceCaption.textContent = meta.caption;
  pieceSource.textContent = meta.source;
  [...galleryHost.querySelectorAll('.seeing-card')].forEach((card) => card.classList.toggle('active', card.dataset.piece === key));
}

function buildTraceObjects(trace, pieceKey) {
  const group = new THREE.Group();
  const states = trace.states.map((q) => stereographicSouth(q));
  const curve = catmullPath(states);

  const ghostPresets = {
    piece1: { opacity: 0.12, radius: 0.118 },
    piece2: { opacity: 0.08, radius: 0.112 },
    piece3: { opacity: 0.03, radius: 0.102 },
    piece4: { opacity: 0.02, radius: 0.098 },
  };
  const ghostPreset = ghostPresets[pieceKey] || ghostPresets.piece1;

  const tubeGeometry = new THREE.TubeGeometry(curve, 220, 0.085, 18, false);
  const tubeMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x1e5cd7,
    roughness: 0.26,
    metalness: 0.08,
    clearcoat: 0.25,
    transparent: true,
    opacity: 0.98,
  });
  const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
  tube.renderOrder = 2;
  group.add(tube);

  const ghostGeometry = new THREE.TubeGeometry(curve, 220, ghostPreset.radius, 12, false);
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: 0x8fb2ee,
    transparent: true,
    opacity: ghostPreset.opacity,
    depthWrite: false,
    side: THREE.BackSide,
  });
  const ghost = new THREE.Mesh(ghostGeometry, ghostMaterial);
  ghost.renderOrder = 0;
  group.add(ghost);

  const pathPoints = curve.getPoints(300);
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x123d8f,
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
  });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  line.renderOrder = 3;
  group.add(line);

  const identitySphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 28, 28),
    new THREE.MeshBasicMaterial({ color: 0x2f8cff, transparent: true, opacity: 0.26, depthWrite: false })
  );
  identitySphere.position.copy(states[0]);
  identitySphere.renderOrder = 1;
  group.add(identitySphere);

  const stateDots = [];
  states.forEach((point, idx) => {
    const isLast = idx === states.length - 1;
    const isInverse = idx === states.length - 2;
    const material = new THREE.MeshStandardMaterial({
      color: isLast ? 0x2c9a62 : isInverse ? 0xd67518 : 0x225fd6,
      emissive: isLast ? 0x1b5e3e : isInverse ? 0x6b3304 : 0x14316f,
      emissiveIntensity: isLast ? 0.82 : 0.34,
      roughness: 0.24,
      metalness: 0.06,
    });
    const dot = new THREE.Mesh(new THREE.SphereGeometry(isLast ? 0.14 : 0.11, 20, 20), material);
    dot.position.copy(point);
    dot.renderOrder = 4;
    group.add(dot);
    stateDots.push(dot);
  });

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 26, 26),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x7ab6ff,
      emissiveIntensity: 1.3,
      roughness: 0.1,
      metalness: 0.05,
    })
  );
  head.renderOrder = 5;
  group.add(head);

  const headHalo = new THREE.Mesh(
    new THREE.SphereGeometry(0.24, 26, 26),
    new THREE.MeshBasicMaterial({ color: 0x8fc5ff, transparent: true, opacity: 0.22, depthWrite: false })
  );
  headHalo.renderOrder = 1;
  group.add(headHalo);

  const closureRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.5, 0.028, 16, 80),
    new THREE.MeshBasicMaterial({ color: 0x3ba96d, transparent: true, opacity: 0.0 })
  );
  closureRing.rotation.x = Math.PI / 2;
  closureRing.position.copy(states[states.length - 1]);
  closureRing.renderOrder = 4;
  group.add(closureRing);

  return { trace, group, states, curve, stateDots, head, headHalo, closureRing };
}

async function main() {
  const [piece1, piece2, piece3, piece4] = await Promise.all([
    fetch(versioned('seeing-code-trace.json')).then((r) => r.json()),
    fetch(versioned('seeing-code-trace-piece2.json')).then((r) => r.json()),
    fetch(versioned('seeing-code-trace-piece3.json')).then((r) => r.json()),
    fetch(versioned('seeing-code-trace-piece4.json')).then((r) => r.json()),
  ]);

  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f7fb);
  scene.fog = new THREE.Fog(0xf5f7fb, 8, 28);

  const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
  camera.position.set(0, 2.2, 10.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 3.4;
  controls.maxDistance = 22;
  controls.target.set(0, 0.35, 0);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;

  const hemi = new THREE.HemisphereLight(0xffffff, 0xdadfe8, 1.35);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(4, 7, 8);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x8fb5ff, 0.6);
  fill.position.set(-6, -2, 4);
  scene.add(fill);

  const pieces = {
    piece1: buildTraceObjects(piece1, 'piece1'),
    piece2: buildTraceObjects(piece2, 'piece2'),
    piece3: buildTraceObjects(piece3, 'piece3'),
    piece4: buildTraceObjects(piece4, 'piece4'),
  };
  Object.values(pieces).forEach((piece) => {
    piece.group.visible = false;
    scene.add(piece.group);
  });
  let activeKey = 'piece1';
  pieces[activeKey].group.visible = true;
  applyPieceText(activeKey);

  const grid = new THREE.GridHelper(16, 10, 0xd9e0ea, 0xe9edf4);
  grid.position.y = -3.8;
  scene.add(grid);

  const clock = new THREE.Clock();

  function buildGallery() {
    const order = ['piece1', 'piece2', 'piece3', 'piece4'];
    galleryHost.innerHTML = '';
    order.forEach((key) => {
      const meta = PIECE_META[key];
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'seeing-card';
      card.dataset.piece = key;
      card.innerHTML = `
        <span class="seeing-card-title">${meta.title}</span>
        <span class="seeing-card-copy">${meta.cardBlurb}</span>
      `;
      card.addEventListener('click', () => {
        if (key === activeKey) return;
        pieces[activeKey].group.visible = false;
        activeKey = key;
        pieces[activeKey].group.visible = true;
        applyPieceText(activeKey);
      });
      galleryHost.appendChild(card);
    });
  }

  buildGallery();

  function resize() {
    const w = canvas.clientWidth || canvas.width;
    const h = canvas.clientHeight || canvas.height;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    controls.update();
  }

  window.addEventListener('resize', resize);
  resize();

  function animate() {
    const elapsed = clock.getElapsedTime();
    const loop = 8.5;
    const phase = (elapsed % loop) / loop;
    const t = phase;
    const active = pieces[activeKey];
    const totalSegments = active.states.length - 1;
    const curvePoint = active.curve.getPoint(t);

    active.head.position.copy(curvePoint);
    active.headHalo.position.copy(curvePoint);
    active.headHalo.scale.setScalar(1 + 0.35 * Math.sin(elapsed * 4.0) ** 2);

    const stepIndex = Math.min(Math.floor(t * totalSegments) + 1, totalSegments);
    metricStep.textContent = `${stepIndex}/${totalSegments}`;
    metricSigma.textContent = Number(active.trace.sigmas[stepIndex]).toFixed(4);
    metricBranch.textContent = active.trace.branches[stepIndex];
    metricProjection.textContent = 'stereo from -1';

    active.stateDots.forEach((dot, idx) => {
      const active = idx === stepIndex;
      dot.scale.setScalar(active ? 1.45 : idx <= stepIndex ? 1.12 : 1.0);
    });

    const closurePhase = Math.max(0, Math.min(1, (phase - 0.74) / 0.18));
    active.closureRing.material.opacity = 0.12 + closurePhase * 0.75;
    active.closureRing.scale.setScalar(1 + 0.45 * closurePhase);

    controls.update();

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  animate();
}

main().catch((err) => {
  console.error(err);
  metricStep.textContent = 'error';
  metricBranch.textContent = 'load failed';
});
