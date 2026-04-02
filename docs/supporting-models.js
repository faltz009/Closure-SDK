const ASSET_VERSION = document.querySelector('meta[name="closure-asset-version"]')?.content || 'dev';
const versioned = (url) => `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(ASSET_VERSION)}`;

const PAPERS = [
  {
    tag: 'Paper 01',
    title: 'Continuous Autoregressive Language Models',
    authors: 'Chenze Shao, Darren Li, Fandong Meng, Jie Zhou',
    arxiv: 'https://arxiv.org/abs/2510.27688',
    image: 'supporting-models-assets/continuousmodels.png',
    abstract: 'This paper tries to get more work out of each prediction step. Instead of choosing one token at a time, it predicts a richer continuous chunk, so long sequences can be generated in fewer moves.',
    relation: 'One Closure state can carry the work of many small symbols. A whole sequence can compile into one closure element, and that same unit can be stored, fetched, and executed as a single object.'
  },
  {
    tag: 'Paper 02',
    title: 'Evolution Strategies at the Hyperscale',
    authors: 'Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, and collaborators',
    arxiv: 'https://arxiv.org/abs/2511.16652',
    image: 'supporting-models-assets/evolution-hyperscale.png',
    abstract: 'This paper is about learning by trying many variations in parallel and keeping the ones that work. Its contribution is to make that style of search fast enough to use on large modern models.',
    relation: 'Search over variants is already built into the substrate. DNA can snapshot and restore state, the VM can rerun programs, and the same representation is used for storing, executing, and comparing candidate improvements.'
  },
  {
    tag: 'Paper 03',
    title: 'TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate',
    authors: 'Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni',
    arxiv: 'https://arxiv.org/abs/2504.19874',
    image: 'supporting-models-assets/turboquant.png',
    abstract: 'This paper is about shrinking vectors so they take less space while still giving trustworthy search results. The key point is that compression can be aggressive without ruining the nearest-neighbor structure the system relies on.',
    relation: 'DNA is already a geometric memory and search depends on distance in that space. If those stored states are ever compressed, retrieval still has to land on the same neighbors, or the memory stops behaving like the same memory.'
  },
  {
    tag: 'Paper 04',
    title: 'LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels',
    authors: 'Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero',
    arxiv: 'https://arxiv.org/abs/2603.19312',
    image: 'supporting-models-assets/world-model.png',
    abstract: 'This paper learns a compact world state directly from pixels and uses it for prediction and planning. Its main value is showing that the state can stay simple enough to work with while still tracking real structure in the world.',
    relation: 'The genome is already a stored world model. It keeps context-keyed transforms in memory, and identity gives the system a built-in notion of expected state versus surprising state, so prediction and deviation are already part of the data structure itself.'
  },
  {
    tag: 'Paper 05',
    title: 'HyperAgents',
    authors: 'Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jeff Clune, Minqi Jiang, Sam Devlin, Tatiana Shavrina',
    arxiv: 'https://arxiv.org/abs/2603.19461',
    image: 'supporting-models-assets/hyperagents.png',
    abstract: 'This paper is about self-improvement that can reach deeper than a fixed outer loop. The central idea is that an agent improves faster when it can also rewrite the part of itself that decides how improvement happens.',
    relation: 'A program can be stored, changed, snapshotted, rolled back, and run again inside the same system. That is the practical shape you need before self-improvement stops being an outer script and becomes part of the computer itself.'
  },
  {
    tag: 'Paper 06',
    title: 'Sheaf theory: from deep geometry to deep learning',
    authors: 'Anton Ayzenberg, German Magai, Thomas Gebhart, Grigory Solomadin',
    arxiv: 'https://arxiv.org/abs/2502.15476',
    image: 'supporting-models-assets/sheaf-learning.png',
    abstract: 'This paper studies a basic question that shows up everywhere in complex systems: when do many local pieces really fit together into one coherent whole? Sheaf theory gives a clean language for agreement, mismatch, and reconstruction across many partial views.',
    relation: 'Table identity already turns many local records into one global state. Search, repair, and execution all depend on tracing how local pieces agree or conflict, which is exactly the local-to-global problem this paper is about.'
  }
];

const TRANSITION_OUT_MS = 180;
const TRANSITION_IN_MS = 340;

let index = 0;
let locked = false;

const deck = document.getElementById('supporting-deck');
const dotsHost = document.getElementById('supporting-dots');
const els = {
  tag: document.getElementById('supporting-tag'),
  counter: document.getElementById('supporting-counter'),
  title: document.getElementById('supporting-title'),
  authors: document.getElementById('supporting-authors'),
  arxiv: document.getElementById('supporting-arxiv'),
  poster: document.getElementById('supporting-poster-link'),
  abstract: document.getElementById('supporting-abstract'),
  relation: document.getElementById('supporting-relation'),
  image: document.getElementById('supporting-image'),
  prev: document.getElementById('supporting-prev'),
  next: document.getElementById('supporting-next')
};

function buildDots() {
  dotsHost.innerHTML = '';
  PAPERS.forEach((paper, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'supporting-dot';
    btn.setAttribute('aria-label', paper.title);
    btn.addEventListener('click', () => goTo(i));
    dotsHost.appendChild(btn);
  });
}

function syncDots() {
  [...dotsHost.children].forEach((dot, i) => {
    dot.classList.toggle('active', i === index);
  });
}

function renderPaper(i) {
  const paper = PAPERS[i];
  index = i;
  els.tag.textContent = paper.tag;
  els.counter.textContent = `${i + 1} / ${PAPERS.length}`;
  els.title.textContent = paper.title;
  els.authors.textContent = paper.authors;
  els.abstract.textContent = paper.abstract;
  els.relation.textContent = paper.relation;
  els.arxiv.href = paper.arxiv;
  els.poster.href = paper.arxiv;
  els.image.src = versioned(paper.image);
  els.image.alt = `${paper.title} first page`;
  els.prev.disabled = i === 0;
  els.next.disabled = i === PAPERS.length - 1;
  syncDots();
}

function goTo(nextIndex) {
  if (locked || nextIndex < 0 || nextIndex >= PAPERS.length || nextIndex === index) {
    return;
  }

  locked = true;
  deck.classList.add('is-leaving');

  window.setTimeout(() => {
    renderPaper(nextIndex);
    deck.classList.remove('is-leaving');
    deck.classList.add('is-entering');

    requestAnimationFrame(() => {
      deck.classList.add('is-entering-active');
    });

    window.setTimeout(() => {
      deck.classList.remove('is-entering', 'is-entering-active');
      locked = false;
    }, TRANSITION_IN_MS);
  }, TRANSITION_OUT_MS);
}

els.prev.addEventListener('click', () => goTo(index - 1));
els.next.addEventListener('click', () => goTo(index + 1));

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowLeft') goTo(index - 1);
  if (event.key === 'ArrowRight') goTo(index + 1);
});

buildDots();
renderPaper(0);
