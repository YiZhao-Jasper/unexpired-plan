'use strict';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

// One shared clock for the matched clips. Source footage and playback rates agree.
const off = $('#off-video');
const on = $('#on-video');
const pairButton = $('#pair-play');
const pairSeek = $('#pair-seek');
let pairPlaying = false;
let userPaused = false;
let inView = false;
let animationFrame = 0;
let lastCorrection = 0;
const safePlay = async (video) => {
  try { await video.play(); return true; } catch (_) { return false; }
};
function paintPair() {
  const t = Math.min(off.currentTime || 0, 12.96);
  if (document.activeElement !== pairSeek) pairSeek.value = t;
  $('#pair-time').textContent = `00:${String(Math.floor(t)).padStart(2, '0')} / 00:13`;
  $('#off-state').textContent = t < 4.2 ? 'Same reference schedule' : 'Biased proposal continues';
  $('#on-state').textContent = t < 4.4 ? 'Reference plan stored' : 'Veto → persistent demotion';
  $('#off-outcome').classList.toggle('visible', t >= 9.5);
  $('#on-outcome').classList.toggle('visible', t >= 9.5);
}
function syncPair(now) {
  if (!pairPlaying) return;
  if (now - lastCorrection > 400 && Math.abs(on.currentTime - off.currentTime) > .08 && on.readyState >= 2) {
    on.currentTime = off.currentTime;
    lastCorrection = now;
  }
  paintPair();
  animationFrame = requestAnimationFrame(syncPair);
}
async function playPair() {
  if (off.readyState >= 1 && off.currentTime >= off.duration - .1) { off.currentTime = 0; on.currentTime = 0; }
  const results = await Promise.all([safePlay(off), safePlay(on)]);
  pairPlaying = results.every(Boolean);
  if (!pairPlaying) { off.pause(); on.pause(); }
  pairButton.textContent = pairPlaying ? 'Ⅱ' : '▶';
  pairButton.setAttribute('aria-label', pairPlaying ? 'Pause comparison' : 'Play comparison');
  cancelAnimationFrame(animationFrame);
  if (pairPlaying) animationFrame = requestAnimationFrame(syncPair);
}
function pausePair() {
  pairPlaying = false;
  off.pause(); on.pause();
  cancelAnimationFrame(animationFrame);
  pairButton.textContent = '▶';
  pairButton.setAttribute('aria-label', 'Play comparison');
}
pairButton.addEventListener('click', () => {
  if (pairPlaying) { userPaused = true; pausePair(); } else { userPaused = false; playPair(); }
});
$('#pair-replay').addEventListener('click', () => { off.currentTime = 0; on.currentTime = 0; userPaused = false; paintPair(); playPair(); });
pairSeek.addEventListener('input', () => {
  const t = Number(pairSeek.value);
  off.currentTime = t; on.currentTime = t;
  paintPair();
});
off.addEventListener('seeked', paintPair);
off.addEventListener('ended', () => {
  pausePair(); paintPair();
  // Keep the outcome visible before restarting the illustrative loop.
  window.setTimeout(() => {
    if (inView && !userPaused && !reducedMotion.matches && document.visibilityState === 'visible') {
      off.currentTime = 0; on.currentTime = 0; playPair();
    }
  }, 1800);
});
const pairObserver = new IntersectionObserver(([entry]) => {
  inView = entry.isIntersecting;
  if (inView && !userPaused && !reducedMotion.matches && document.visibilityState === 'visible') playPair();
  else pausePair();
}, { threshold: .18 });
pairObserver.observe($('#comparison'));
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') pausePair();
  else if (inView && !userPaused && !reducedMotion.matches) playPair();
});
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) pausePair(); });

// Show only the two actual paper frames: no generated intermediate rollout.
$$('[data-phase]').filter(el => el.tagName === 'BUTTON').forEach(button => {
  button.addEventListener('click', () => {
    const phase = button.dataset.phase;
    $$('.segmented button[data-phase]').forEach(item => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-pressed', String(item === button));
    });
    $('.evidence-grid').dataset.phase = phase;
    $$('.paper-frame img').forEach((img, i) => {
      img.src = img.dataset[phase];
      const labels = ['Unaccelerated reference', 'Benign acceleration', 'Harmful acceleration'];
      img.alt = `${labels[i]}: original Figure 1 frame at ${phase}`;
    });
  });
});

// Temporal alignment example: H=8, execution horizon=2, refresh every 4 calls.
const callStep = $('#call-step');

// Start the method demonstrations on arrival. Manual choices pause their own
// loop; hidden tabs suspend timers without losing the visitor's playback choice.
function automaticSequence({ button, output, delay, advance, label, playText, pauseText }) {
  let playing = !reducedMotion.matches;
  let timer = null;
  function sync() {
    clearInterval(timer);
    timer = null;
    button.innerHTML = `${playing ? pauseText : playText} <span aria-hidden="true">${playing ? 'Ⅱ' : '▶'}</span>`;
    button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${label}`);
    output.setAttribute('aria-live', playing ? 'off' : 'polite');
    if (playing && document.visibilityState === 'visible') timer = setInterval(advance, delay);
  }
  button.addEventListener('click', () => { playing = !playing; sync(); });
  document.addEventListener('visibilitychange', sync);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) { playing = false; sync(); }
  });
  sync();
  return { pause() { playing = false; sync(); } };
}

function drawPlan() {
  const call = Number(callStep.value);
  const base = call === 4 ? 8 : 0;
  const start = call * 2;
  const isReference = call === 0 || call === 4;
  $('#reference-plan').replaceChildren();
  $('#proposal-plan').replaceChildren();
  for (let i = 0; i < 10; i++) {
    const index = base + i;
    const cell = document.createElement('span');
    cell.className = 'action-cell';
    if (i >= 8) { cell.classList.add('empty'); cell.setAttribute('aria-hidden', 'true'); }
    else {
      cell.textContent = index;
      if (index < start) cell.classList.add('past');
      if (index >= start && index < start + 2) cell.classList.add('compared');
    }
    $('#reference-plan').append(cell);
    const proposed = document.createElement('span');
    proposed.className = 'action-cell';
    // Only the next executed prefix is drawn; indices share the same columns.
    if (index >= start && index < start + 2) {
      proposed.textContent = index;
      proposed.classList.add('compared');
    } else { proposed.classList.add('empty'); proposed.setAttribute('aria-hidden', 'true'); }
    $('#proposal-plan').append(proposed);
  }
  $('#reference-label').textContent = call === 4 ? 'New reference plan' : 'Saved reference';
  $('#proposal-label').textContent = isReference ? `Reference call ${call}` : `Accelerated call ${call}`;
  $('#alignment-note').textContent = isReference ? `Execute indices ${start}–${start + 1}` : `Compare indices ${start}–${start + 1}`;
  $('#call-output').textContent = call;
  $('#call-explanation').textContent = call === 0
    ? 'Compute one full reference plan. Execute actions 0 and 1; save the remaining actions for later checks.'
    : call === 4
      ? 'The old reference has expired. Run the next scheduled reference call, execute actions 8 and 9, and store a new plan.'
      : `Read actions ${start} and ${start + 1} from the saved reference. No new reference forward pass is needed for this check.`;
}
callStep.value = 0;
drawPlan();
const planSequence = automaticSequence({
  button: $('#plan-play'), output: $('#call-explanation'), delay: 1900,
  label: 'alignment sequence', playText: 'Play sequence', pauseText: 'Pause sequence',
  advance() { callStep.value = (Number(callStep.value) + 1) % 5; drawPlan(); }
});
callStep.addEventListener('input', () => { planSequence.pause(); drawPlan(); });

const checks = {
  agree: { path: 'M25 114C160 39 289 10 495 72', text: 'Small deviation δ and no near repetition: let the accelerated action through.', title: 'Schematic action curves: reference and accelerated proposal agree' },
  drift: { path: 'M25 117C160 92 289 85 495 123', text: 'Large deviation δ: veto the proposal and execute the aligned stored reference slice.', title: 'Schematic action curves: the accelerated proposal drifts away from the reference' },
  repeat: { path: 'M25 117C160 42 289 13 495 75', text: 'Re-emitting the saved plan can make δ vanish. The near-repetition check λ supplies the veto that deviation alone can miss.', title: 'Schematic action curves: repetition agrees with the old plan but can miss what a fresh reference would say' }
};
const checkKinds = Object.keys(checks);
let currentCheck = 0;
function drawCheck(kind) {
  currentCheck = checkKinds.indexOf(kind);
  $$('[data-check]').forEach(item => { item.classList.toggle('active', item.dataset.check === kind); item.setAttribute('aria-pressed', String(item.dataset.check === kind)); });
  $('#proposal-curve').setAttribute('d', checks[kind].path);
  $('#proposal-curve').setAttribute('stroke-dasharray', kind === 'repeat' ? '10 9' : 'none');
  $('#truth-curve').style.opacity = kind === 'repeat' ? 1 : 0;
  $('#truth-key').hidden = kind !== 'repeat';
  $('#check-explanation').textContent = checks[kind].text;
  $('#check-svg-title').textContent = checks[kind].title;
}
const checkSequence = automaticSequence({
  button: $('#checks-play'), output: $('#check-explanation'), delay: 2800,
  label: 'monitor cases', playText: 'Play', pauseText: 'Pause',
  advance() { drawCheck(checkKinds[(currentCheck + 1) % checkKinds.length]); }
});
$$('[data-check]').forEach(button => button.addEventListener('click', () => {
  checkSequence.pause();
  drawCheck(button.dataset.check);
}));

const figureDialog = $('#figure-dialog');
const figureDescriptions = {
  evidence: 'Figure 1 · A gate can accept a harmful shortcut',
  method: 'Figure 2 · The unexpired plan monitor'
};
$$('[data-figure]').forEach(button => button.addEventListener('click', () => {
  const kind = button.dataset.figure;
  $('#enlarged-figure').src = `assets/figures/${kind}.svg`;
  $('#enlarged-figure').alt = figureDescriptions[kind];
  $('#figure-title').textContent = figureDescriptions[kind];
  $('#figure-download').href = `assets/figures/${kind}.svg`;
  figureDialog.showModal();
}));
$('#close-figure').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('click', event => {
  if (event.target !== figureDialog) return;
  const rect = figureDialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) figureDialog.close();
});

const film = $('#film-video');
$$('[data-time]').forEach(button => button.addEventListener('click', () => {
  const seek = () => { film.currentTime = Number(button.dataset.time); safePlay(film); };
  if (film.readyState >= 1) seek();
  else { film.addEventListener('loadedmetadata', seek, { once: true }); film.load(); }
  film.scrollIntoView({ block: 'center', behavior: reducedMotion.matches ? 'instant' : 'smooth' });
}));
film.addEventListener('play', () => { pausePair(); planSequence.pause(); checkSequence.pause(); });

$('#copy-citation').addEventListener('click', async () => {
  let copied = false;
  try { await navigator.clipboard.writeText($('#bibtex').textContent); copied = true; } catch (_) {
    const range = document.createRange(); range.selectNodeContents($('#bibtex'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
  }
  $('#copy-status').textContent = copied ? 'Citation copied.' : 'Citation selected. Use your browser copy command.';
  $('#copy-citation').textContent = copied ? 'Copied ✓' : 'Selected — copy text';
  setTimeout(() => { $('#copy-citation').innerHTML = 'Copy BibTeX <span aria-hidden="true">↗</span>'; }, 2500);
});
const navObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    $$('.site-header nav a').forEach(link => link.classList.toggle('current', link.hash === '#' + entry.target.id));
  });
}, { rootMargin: '-15% 0px -65% 0px' });
['idea', 'method', 'results', 'film'].forEach(id => navObserver.observe(document.getElementById(id)));
