const $ = id => document.getElementById(id);
const num = v => typeof v === "number";
const fmt = (v,d) => num(v) ? v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}) : v;
const backends = ['HIP', 'Vulkan'], caches = ['f16/f16', 'q8_0/q8_0', 'q4_0/q4_0'];
const cacheName = value => value.replaceAll('q4_0', 'Q4').replaceAll('q8_0', 'Q8');
function segments(id, values, selected, key) {
  $(id).innerHTML = values.map(v => `<button aria-pressed="${v === selected}" data-filter="${key}" data-value="${v}">${v}</button>`).join('');
}
const benchData = window.BENCHMARK_DATA.models;
const families = {
  gemma12: 'Gemma 12B', 'gemma12-qat': 'Gemma 12B', 'gemma-e4b': 'Gemma E4B',
  qwen35: 'Qwen 9B'
};
const baseModel = row => families[row.family] || row.family;
const filters = {model:'All', weights:'All', backend:'All', cache:'All'};
$('bench-image').replaceChildren(...benchData.rubric.map(i=>new Option(i.id,i.id)));

function showBenchImage() {
  const row = benchData.rows.find(r=>r.id === $('bench-config').value);
  const image = benchData.rubric.find(i=>i.id === $('bench-image').value);
  const sample = row?.images[image.id];
  const answer = benchData.answers[sample?.answer_id];
  $('bench-picture').src = '../media/qengineering/' + image.file;
  $('bench-picture').alt = image.id;
  $('bench-question').textContent = image.prompt;
  $('bench-answer').textContent = answer?.text || sample?.error || 'No saved answer for this configuration';
  $('bench-review').textContent = [num(sample?.quality) ? fmt(sample.quality,2)+'% image quality' : '', answer?.review?.note].filter(Boolean).join(' · ');
  $('bench-image-metrics').textContent = sample ? [
    num(sample.generation_ts) ? fmt(sample.generation_ts,2)+' generation t/s' : '',
    num(sample.seconds) ? fmt(sample.seconds,2)+' seconds' : '',
    sample.finish_reason ? 'Finish: '+sample.finish_reason : '',
    num(sample.coverage_score) ? 'Keyword coverage: '+fmt(sample.coverage_score,2)+'% (not quality)' : ''
  ].filter(Boolean).join(' · ') : 'No image measurement';
  const review = answer?.review;
  $('bench-score-breakdown').textContent = review ? [
    ['Accuracy', 'correctness', 40], ['Visible detail', 'visual_detail', 30],
    ['Completeness', 'completeness', 20], ['Clarity', 'clarity', 10]
  ].filter(([,key]) => num(review[key])).map(([label,key,max]) => `${label}: ${review[key]}/${max}`).join(' · ') : '';
  $('bench-reference').textContent = benchData.reference.answers[image.id].text;
}

function renderBench() {
  segments('bench-model',['All',...new Set(benchData.rows.map(baseModel))],filters.model,'model');
  const modelRows = benchData.rows.filter(r => filters.model === 'All' || baseModel(r) === filters.model);
  const quants = [...new Set(modelRows.map(r => r.quant))].sort();
  if (!quants.includes(filters.weights)) filters.weights = 'All';
  segments('bench-weights',['All',...quants],filters.weights,'weights');
  segments('bench-backend',['All',...backends],filters.backend,'backend');
  segments('bench-cache',['All',...caches.map(cacheName)],filters.cache,'cache');
  const rows = modelRows.filter(r =>
    (filters.weights === 'All' || r.quant === filters.weights) &&
    (filters.backend === 'All' || r.backend === filters.backend.toLowerCase()) &&
    (filters.cache === 'All' || cacheName(r.cache+'/'+r.cache) === filters.cache));
  const benchRows = rows;
  const previous = $('bench-config').value;
  $('bench-config').replaceChildren(...rows.map(r=>new Option(r.id,r.id)));
  if (rows.some(r=>r.id===previous)) $('bench-config').value = previous;
  $('bench-preview').hidden = !rows.length;
  if (rows.length) showBenchImage();
  $('bench-rows').innerHTML = [...new Set(benchRows.map(r => r.model))]
    .flatMap(id => caches.flatMap(cache => backends.flatMap(backend =>
      benchRows.filter(r => r.model === id && r.cache+'/'+r.cache === cache && r.backend === backend.toLowerCase())
        .map(benchmarkRow)))).join('');
}

function imageQuality(row) {
  const scores = benchData.rubric.map(image => row.images[image.id]?.quality);
  return scores.every(num) ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
}

function benchmarkRow(row) {
  const speed = key => num(row[key])
    ? fmt(row[key],2) + (num(row[key+'_sd']) ? ' ± '+fmt(row[key+'_sd'],2) : '')
    : 'Not recorded';
  const quality = imageQuality(row);
  return `<tr>
    <td>${baseModel(row)}</td>
    <td class="${row.backend}">${row.backend === 'hip' ? 'HIP' : 'Vulkan'}</td>
    <td>${row.quant.replace('QAT_', 'QAT ')}</td>
    <td>${cacheName(row.cache+'/'+row.cache)}</td>
    <td>${speed('pp512')}</td>
    <td>${speed('tg128')}</td>
    <td>${num(quality) ? fmt(quality,2) : 'Unscored'}</td>
    <td>${num(row.memory_mib) ? fmt(row.memory_mib/1024,2) : 'Not recorded'}</td>
    <td><button type="button" data-benchrow="${row.id}">View</button></td>
  </tr>`;
}
$('bench-config').onchange = showBenchImage;
$('bench-image').onchange = showBenchImage;


$('bench-panel').addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.benchrow) {
    $('bench-config').value = button.dataset.benchrow;
    showBenchImage();
    $('bench-preview').open = true;
    $('bench-config').focus();
    return;
  }
  const {filter, value} = button.dataset;
  if (Object.hasOwn(filters, filter)) {
    filters[filter] = value;
    renderBench();
    $('bench-'+filter).querySelector('[aria-pressed="true"]').focus({preventScroll:true});
  }
});
renderBench();
