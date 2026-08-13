// Detections whose confidence falls below this get flagged for a second look.
// It is the number a reviewer should spend attention on; everything else the
// system is already sure about.
const REVIEW_CONFIDENCE = 0.75;

const SAMPLES = [
  { file: "uniform_residential_appraisal.png", title: "Uniform Residential", subtitle: "Freddie Mac 70, ~129 checkboxes" },
  { file: "manufactured_home_appraisal.png",  title: "Manufactured Home",  subtitle: "Freddie Mac 70B, watermarked scan" },
  { file: "market_conditions_addendum.png",   title: "Market Conditions",  subtitle: "Fannie Mae 1004MC" },
  { file: "neighborhood_site_section.png",    title: "Neighborhood",       subtitle: "Neighborhood + site fields" },
];

const state = { boxes: [], filter: 'all', active: null, image: null };

const els = {
  file:      document.getElementById('file'),
  drop:      document.getElementById('drop'),
  hero:      document.getElementById('hero'),
  loading:   document.getElementById('loadingState'),
  stage:     document.getElementById('stage'),
  list:      document.getElementById('list'),
  listCard:  document.getElementById('listCard'),
  summary:   document.getElementById('summary'),
  errorBox:  document.getElementById('error'),
  hero_grid: document.getElementById('samplesHero'),
  strip:     document.getElementById('samplesStrip'),
  stripCard: document.getElementById('samplesStripCard'),
  loadLbl:   document.getElementById('loadingLabel'),
};

// -- Sample cards ---------------------------------------------------------

for (const sample of SAMPLES) {
  const card = document.createElement('button');
  card.className = 'sample-card';
  card.type = 'button';
  card.innerHTML = `
    <div class="sample-thumb" style="background-image: url('/samples/${sample.file}')"></div>
    <div class="sample-label"><b>${sample.title}</b><span>${sample.subtitle}</span></div>`;
  card.addEventListener('click', () => loadSample(sample.file, sample.title));
  els.hero_grid.appendChild(card);
}
for (const sample of SAMPLES) {
  const chip = document.createElement('button');
  chip.className = 'chip'; chip.type = 'button'; chip.textContent = sample.title;
  chip.addEventListener('click', () => loadSample(sample.file, sample.title));
  els.strip.appendChild(chip);
}

async function loadSample(name, title) {
  els.loadLbl.textContent = `Loading ${title}...`;
  const response = await fetch('/samples/' + name);
  analyse(new File([await response.blob()], name, { type: 'image/png' }));
}

// -- File input & drag-and-drop ------------------------------------------

els.file.addEventListener('change', () => {
  if (els.file.files.length) analyse(els.file.files[0]);
});
['dragenter', 'dragover'].forEach(evt =>
  els.drop.addEventListener(evt, e => { e.preventDefault(); els.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(evt =>
  els.drop.addEventListener(evt, e => { e.preventDefault(); els.drop.classList.remove('over'); }));
els.drop.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) analyse(file);
});

// -- Filter buttons -------------------------------------------------------

document.querySelectorAll('[data-filter]').forEach(button =>
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-filter]').forEach(o => o.classList.remove('on'));
    button.classList.add('on');
    state.filter = button.dataset.filter;
    renderList(); draw();
  }));

// -- Analyse --------------------------------------------------------------

function showError(message) {
  els.errorBox.innerHTML = `<div class="error">${message}</div>`;
}

function showLoading(label) {
  els.errorBox.innerHTML = '';
  els.hero.hidden = true;
  els.stage.hidden = true;
  els.loading.hidden = false;
  els.loadLbl.textContent = label || 'Reading the page...';
}

function showStage() {
  els.hero.hidden = true;
  els.loading.hidden = true;
  els.stage.hidden = false;
  els.stripCard.hidden = false;
}

async function analyse(file) {
  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    return analysePdf(file);
  }
  showLoading('Detecting checkboxes...');
  els.summary.hidden = true;
  els.listCard.hidden = true;

  const body = new FormData();
  body.append('file', file);

  let payload;
  try {
    const response = await fetch('/detect/debug', { method: 'POST', body });
    payload = await response.json();
    if (!response.ok) {
      els.loading.hidden = true;
      els.hero.hidden = false;
      showError(payload.detail || 'The page could not be processed.');
      return;
    }
  } catch (err) {
    els.loading.hidden = true;
    els.hero.hidden = false;
    showError('Could not reach the service.');
    return;
  }

  state.boxes = payload.boxes.map((box, index) => ({ ...box, index }));
  state.active = null;

  // Original image displayed from the local file; overlay drawn on a canvas
  // above it, so filtering costs no round trip.
  const image = new Image();
  image.onload = () => {
    els.stage.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'canvas-wrap';
    const canvas = document.createElement('canvas');
    wrap.append(image, canvas);
    els.stage.appendChild(wrap);
    state.image = { element: image, canvas };
    showStage();
    draw();
  };
  image.src = URL.createObjectURL(file);
  image.alt = 'Appraisal page with detected checkboxes outlined';

  renderSummary(payload);
  renderList();
}

async function analysePdf(file) {
  showLoading('Rasterising and detecting across every page...');
  els.summary.hidden = true;
  els.listCard.hidden = true;

  const body = new FormData();
  body.append('file', file);

  let payload;
  try {
    const response = await fetch('/detect/pdf', { method: 'POST', body });
    payload = await response.json();
    if (!response.ok) {
      els.loading.hidden = true;
      els.hero.hidden = false;
      showError(payload.detail || 'The document could not be processed.');
      return;
    }
  } catch (err) {
    els.loading.hidden = true;
    els.hero.hidden = false;
    showError('Could not reach the service.');
    return;
  }

  const checked = payload.pages.reduce(
    (total, page) => total + page.boxes.filter(box => box.is_checked).length, 0);

  els.stage.innerHTML = `<div class="loading-hero" style="max-width:520px">
      <p style="font-size:15px; color:var(--text); margin-bottom:6px"><b>${payload.page_count} pages processed</b></p>
      <p>${payload.total_boxes} checkboxes found, ${checked} marked.</p>
      <small>Rendered at ${payload.render_dpi} DPI in ${payload.elapsed_ms} ms.<br>Upload a single page image to inspect detections visually.</small>
    </div>`;
  showStage();

  document.getElementById('tChecked').textContent = checked;
  document.getElementById('tUnchecked').textContent = payload.total_boxes - checked;
  document.getElementById('tReview').textContent = '-';
  document.getElementById('reviewBanner').hidden = true;
  els.summary.hidden = false;
}

// -- Rendering ------------------------------------------------------------

function needsReview(box) { return box.confidence < REVIEW_CONFIDENCE; }

function visibleBoxes() {
  if (state.filter === 'checked')   return state.boxes.filter(box => box.is_checked);
  if (state.filter === 'unchecked') return state.boxes.filter(box => !box.is_checked);
  if (state.filter === 'review')    return state.boxes.filter(needsReview);
  return state.boxes;
}

function renderSummary(payload) {
  const flagged = state.boxes.filter(needsReview).length;
  document.getElementById('tChecked').textContent = payload.checked_count;
  document.getElementById('tUnchecked').textContent = payload.unchecked_count;
  document.getElementById('tReview').textContent = flagged;
  document.getElementById('reviewCount').textContent = flagged;
  document.getElementById('reviewTotal').textContent = state.boxes.length;
  document.getElementById('reviewBanner').hidden = flagged === 0;
  els.summary.hidden = false;
}

function renderList() {
  const boxes = visibleBoxes();
  els.listCard.hidden = state.boxes.length === 0;

  els.list.innerHTML = boxes.map(box => {
    const colour = box.is_checked ? 'var(--checked)' : 'var(--unchecked)';
    const flag = needsReview(box) ? '<span class="flag">review</span>' : '';
    return `
      <div class="row ${state.active === box.index ? 'active' : ''}" data-index="${box.index}">
        <span class="swatch" style="background:${colour}"></span>
        <span>
          ${box.is_checked ? 'Checked' : 'Unchecked'}
          <span class="meta">&middot; ink ${box.ink_ratio.toFixed(3)} &middot; conf ${box.confidence.toFixed(2)}</span>
        </span>
        ${flag}
      </div>`;
  }).join('') || '<div class="pad" style="color:var(--muted); font-size:13px; text-align:center">No detections in this filter.</div>';

  els.list.querySelectorAll('.row').forEach(row =>
    row.addEventListener('click', () => {
      state.active = Number(row.dataset.index);
      renderList(); draw();
    }));
}

function draw() {
  if (!state.image) return;
  const { element, canvas } = state.image;

  const rect = element.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;

  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const scale = rect.width / element.naturalWidth;
  const styles = getComputedStyle(document.documentElement);
  const palette = {
    checked:   styles.getPropertyValue('--checked').trim(),
    unchecked: styles.getPropertyValue('--unchecked').trim(),
    review:    styles.getPropertyValue('--review').trim(),
    indigo:    styles.getPropertyValue('--indigo').trim(),
  };

  for (const box of visibleBoxes()) {
    const [x1, y1, x2, y2] = box.bbox;
    const isActive = state.active === box.index;
    context.strokeStyle = isActive
      ? palette.indigo
      : needsReview(box) ? palette.review
      : box.is_checked ? palette.checked : palette.unchecked;
    context.lineWidth = isActive ? 3 : 1.8;
    context.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);

    if (isActive) {
      context.fillStyle = palette.indigo + '22';
      context.fillRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    }
  }
}

window.addEventListener('resize', draw);
