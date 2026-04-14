(function () {
  const NAV = `
<nav class="site-nav" aria-label="Site navigation">
  <details class="nav-mobile-details">
    <summary class="nav-mobile-summary">
      <span class="nav-mobile-title">Closure</span>
      <span class="nav-mobile-toggle">Menu</span>
    </summary>
    <div class="nav-mobile-menu">
      <a href="index.html">Home</a>
      <a href="stack.html">Where we are</a>
      <a href="horizon.html">What the future holds</a>
      <a href="origin.html">How we got here</a>
      <a href="supporting-models.html">Supporting Models</a>
      <a class="nav-mobile-accent" href="seeing-code.html">Seeing Code</a>
      <a class="nav-mobile-accent" href="experiments.html">Live Experiments</a>
      <a href="support.html">Support &amp; Collaborate</a>
      <a href="newsletter.html">Newsletter</a>
      <a href="about.html">About</a>
    </div>
  </details>
  <div class="nav-left">
    <a class="nav-home" href="index.html">Closure</a>
    <span class="nav-sep">/</span>
    <span class="nav-triad">
      <a href="stack.html">Where we are</a>
      <span class="nav-dot">&bull;</span>
      <a href="horizon.html">What the future holds</a>
      <span class="nav-dot">&bull;</span>
      <a href="origin.html">How we got here</a>
    </span>
    <span class="nav-sep">/</span>
    <a href="supporting-models.html">Supporting Models</a>
  </div>
  <div class="nav-right">
    <a class="nav-pill" href="experiments.html">Live Experiments</a>
    <span class="nav-dot">&bull;</span>
    <a class="nav-pill" href="seeing-code.html">Seeing Code</a>
    <span class="nav-dot">&bull;</span>
    <a href="support.html">Support &amp; Collaborate</a>
    <span class="nav-dot">&bull;</span>
    <a href="newsletter.html">Newsletter</a>
    <span class="nav-sep">/</span>
    <a href="about.html">About</a>
  </div>
</nav>`;

  const root = document.getElementById('site-nav-root');
  if (!root) return;
  root.outerHTML = NAV;

  const page = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.site-nav a[href="' + page + '"]').forEach(function (a) {
    a.setAttribute('aria-current', 'page');
  });
})();
