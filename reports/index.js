const DATA = window.BENCHMARK_DATA.comparison;
const P = structuredClone(DATA.profiles);
const modelNames = {"gemma12-Q4_0":"Gemma 12B Q4_0", "qwen35-MTP-Q4_K_M":"Qwen 9B Q4_K_M", "gemma-e4b-Q4_0":"Gemma E4B Q4_0"};
const models = Object.values(modelNames);
const backends = ['HIP', 'Vulkan'], caches = ['f16/f16', 'Q8/Q8', 'Q4/Q4'];
const displayCaches = ['f16/f16', 'Q4/Q4', 'Q8/Q8'];
function profileOrder(a,b) {
  return models.indexOf(a[0]) - models.indexOf(b[0])
    || backends.indexOf(a[1]) - backends.indexOf(b[1])
    || displayCaches.indexOf(a[2]) - displayCaches.indexOf(b[2])
    || String(a[9] || '').localeCompare(String(b[9] || ''));
}
const cacheName = value => value.replaceAll('q4_0', 'Q4').replaceAll('q8_0', 'Q8');
const reading = value => value === 'Not scored' ? value : /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : 'Not measured';
for (const p of P) for (const r of p.rows) r[9] = p.c === '16K' && p.u !== 4 ? 'Unified' : 'Separate';
for (const block of Object.values(window.BENCHMARK_DATA.views.mtp)) {
  if (block.kind !== 'rows') continue;
  for (const entry of block.rows) {
    const c = entry.cells.map(cell => cell.value);
    if (c.length !== 16 || !modelNames[benchmarkModelId(c[0])] || !['off','on'].includes(c[3])) continue;
    const p = P.find(p => p.u === Number(c[2]) && p.c === '16K');
    const name = modelNames[benchmarkModelId(c[0])], backend = c[1] === 'hip' ? 'HIP' : 'Vulkan', cache = cacheName(c[4]);
    let r = p.rows.find(r => r[0] === name && r[1] === backend && r[2] === cache && r[9] === 'Unified');
    if (!r) { r = [name,backend,cache,...Array.from({length:5},()=>['Not measured','Not measured']),{},'Unified']; p.rows.push(r); }
    const state = c[3] === 'on' ? 1 : 0;
    const failed = ['failed','oom','timeout','unsupported'].includes(c[15]);
    const values = [reading(c[10]),reading(c[6]),reading(c[7]),reading(c[13]),reading(c[14])].map(v => failed && v === 'Not measured' ? 'Fail' : v);
    values[0] = typeof values[0] === 'number' ? values[0]/1000 : values[0];
    values[2] = typeof values[2] === 'number' ? values[2]*p.u : values[2];
    values.forEach((v,i) => {
      const prior = r[i+3][state];
      if (i===2 && typeof prior === 'number' && typeof v === 'number' && Math.abs(prior-v)<=.005*p.u+.00001) return;
      if (typeof v === 'number' || v === 'Fail' || typeof prior !== 'number') r[i+3][state] = v;
    });
  }
}
const cachedRunGroups = ['cases','reruns','context64','two_user_cases','context256','extension_cases'];
for (const group of cachedRunGroups) {
  for (const c of window.BENCHMARK_DATA.cached?.[group] || []) {
    if (!c.status || c.status === 'pending' || c.kv_unified === true || typeof c.mtp !== 'boolean') continue;
    const name = modelNames[benchmarkModelId(c.model)], backend = backends.find(b => b.toLowerCase() === c.backend?.toLowerCase());
    const p = P.find(p => p.u === c.slots && parseInt(p.c) * 1024 === c.context_per_user);
    if (!name || !backend || !p || !c.cache) continue;
    const cache = cacheName(c.cache + '/' + c.cache);
    let r = p.rows.find(r => r[0] === name && r[1] === backend && r[2] === cache && r[9] === 'Separate');
    if (!r) {
      r = [name,backend,cache,...Array.from({length:5},()=>['Not measured','Not measured']),{},'Separate'];
      const quality = P.find(v => v.u === p.u && v.c === '16K')?.rows.find(v => v[0] === name && v[1] === backend && v[2] === cache && v[9] === 'Unified');
      if (quality) r[6] = [...quality[6]];
      p.rows.push(r);
    }
    const state = c.mtp ? 1 : 0;
    const status = c.status === 'timeout' ? 'Timeout' : c.status === 'oom' ? 'OOM' : 'Fail';
    for (const [column,field] of [[3,'ttft_seconds_mean'],[4,'prompt_tps_per_user_mean'],[5,'decode_aggregate_tps']]) {
      if (c.status !== 'passed') r[column][state] = status;
      else if (typeof c[field] === 'number') r[column][state] = c[field];
    }
    r[7][state] = c.status === 'passed' ? (typeof c.peak_gpu_mib === 'number' ? c.peak_gpu_mib / 1024 : 'Not measured') : status;
    r[8] = {};
  }
}
function plannedProfile(p, model, backend, cache, workload = overviewWorkload) {
  if (workload === 'Short') return p.c === '16K';
  return backends.includes(backend) && cache === (model.includes('E4B') ? 'f16/f16' : 'Q4/Q4')
    && (p.c !== '16K' || p.u === 4) && !(model.includes('E4B') && p.c === '256K');
}
function matchingRows(p, backend = selectedBackend, cache = selectedCache) {
  const pool = overviewWorkload === 'Short' ? 'Unified' : 'Separate';
  const rows = p.rows.filter(r => r[9] === pool && (includeQ8 || r[2] !== 'Q8/Q8') && (backend === 'All' || r[1] === backend) && (cache === 'All' || r[2] === cache));
  for (const model of models) for (const b of backends) for (const k of caches) {
    if ((!includeQ8 && k === 'Q8/Q8') || (backend !== 'All' && backend !== b) || (cache !== 'All' && cache !== k)) continue;
    if (plannedProfile(p,model,b,k) && !rows.some(r => r[0] === model && r[1] === b && r[2] === k)) {
      rows.push([model,b,k,...Array.from({length:5},()=>['Not measured','Not measured']),{},pool]);
    }
  }
  return rows.sort(profileOrder);
}
// idx = position in row array; better: +1 higher is better, -1 lower is better; mode: "%" relative or "pp"/"abs" absolute
const M = [
 {i:5,name:"Speed",unit:"t/s",better:1,mode:"%",d:2},
 {i:4,name:"Prefill",unit:"t/s/user",better:1,mode:"%",d:2},
 {i:3,name:"TTFT",unit:"s, mean",better:-1,mode:"%",d:2},
 {i:6,name:"Image quality",unit:"% at 16K",better:1,mode:"pp",d:2},
 {i:7,name:"Peak GPU",unit:"GiB",better:-1,mode:"abs",d:2},
];

const num = v => typeof v === "number";
function delta(m, [a,b]){
  if ([a,b].some(v => ['Not measured','Not supported'].includes(v))) return {txt: [a,b].find(v => ['Not measured','Not supported'].includes(v)), cls:'neutral', v:null};
  if(a === "Not scored" || b === "Not scored") return {txt:"Not scored", cls:"neutral", v:null};
  if(!num(a)||!num(b)) return {txt: num(a)&&!num(b) ? b : "Fail", cls:"bad", v:null};
  const raw = m.mode==="%" ? (b-a)/a*100 : b-a;
  const sign = raw>0?"+":raw<0?"−":"±";
  const unit = m.mode==="%"?"%":m.mode==="pp"?" pp":" GiB";
  const thr = m.mode==="%"?2:m.mode==="pp"?0.5:0.25;
  const cls = Math.abs(raw)<thr?"neutral":(raw*m.better>0?"good":"bad");
  return {txt:sign+Math.abs(raw).toFixed(m.mode==="%"?1:2)+unit, cls, v:raw*m.better};
}
const fmt = (v,d) => num(v) ? v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}) : v;

let includeQ8 = false, overviewWorkload = 'Short';
let sel = 0, heatMetric = 0, selectedBackend = 'All', selectedCache = 'All';
const $ = id => document.getElementById(id);

function bar(cls, v, max, d){
  if(!num(v)) return `<div class="bar ${cls}" aria-label="MTP ${cls}: ${v}"><span class="${['Not scored','Not measured','Not supported'].includes(v) ? 'neutral' : 'fail'} bar-status">${v}</span></div>`;
  return `<div class="bar ${cls}" aria-label="MTP ${cls}: ${v}"><i style="width:calc((100% - 58px) * ${v/max})"></i><span>${fmt(v,d)}</span></div>`;
}

function renderCards(){
  const p = P[sel];
  const profiles = matchingRows(p);
  const missing = profiles.flatMap(r => [0,1].filter(state => r[5][state] === 'Not measured'));
  $("note").textContent = profiles.length
    ? `${overviewWorkload === 'Short' ? 'Real prompts · Short prompt:' : 'Real prompts · Long prompt already cached;'} 512 new tokens, 128 output tokens${missing.length ? ` · ${missing.length} MTP state${missing.length === 1 ? '' : 's'} not measured` : ''}`
    : 'Outside scope: this workload was not planned for the selected settings';
  $("cards").innerHTML = profiles.map(r=>{
    if (r.slice(3,8).every(pair => pair.every(v => ['Not measured','Not supported'].includes(v)))) return `<article class="card"><div class="card-head"><div><h3 class="card-title">${r[0]}</h3>${chips([p.u+' user'+(p.u === 1 ? '' : 's'), p.c+'/user', r[1], 'KV '+r[2], r[9]+' KV'])}</div></div><p class="small">${r[3][0]}</p></article>`;
    const missingStates = ['off','on'].filter((state,i) => r[5][i] === 'Not measured');
    const missingNotice = missingStates.length ? `<p class="missing-result">MTP ${missingStates.join(' / ')}: Not measured</p>` : '';
    const sp = rowDelta(M[0], r);
    const rows = M.map(m=>{
      const vals = r[m.i], nums = vals.filter(num);
      const max = nums.length ? Math.max(...nums) : 1;
      const dl = rowDelta(m, r);
      return `<div class="row"><div class="lbl">${m.name}<small>${m.unit}</small></div>
        <div class="bars">${bar("off",vals[0],max,m.d)}${bar("on",vals[1],max,m.d)}</div>
        <div class="delta ${dl.cls}">${dl.txt}</div></div>`;
    }).join("");
    return `<article class="card"><div class="card-head">
      <div><h3 class="card-title">${r[0]}</h3>${chips([p.u+' user'+(p.u === 1 ? '' : 's'), p.c+'/user', r[1], 'KV '+r[2], r[9]+' KV'])}</div>
      <div class="verdict">speed with MTP<b class="${sp.cls}">${sp.txt}</b></div></div>${missingNotice}${rows}${timingDetails(p,r)}</article>`;
  }).join("");
}

function segments(id, values, selected, key) {
  $(id).innerHTML = values.map(v => `<button aria-pressed="${v === selected}" data-${key}="${v}">${v}</button>`).join('');
}
function renderPicker(){
  $('seg-workload').innerHTML = ['Short','Long'].map(key =>
    `<button aria-pressed="${key===overviewWorkload}" data-workload="${key}">${key} prompt</button>`).join('');

  segments('seg-backend',['All',...backends],selectedBackend,'backend');
  segments('seg-cache',['All',...displayCaches.filter(c => includeQ8 || c !== 'Q8/Q8')],selectedCache,'cache');
  const p = P[sel];
  $("seg-users").innerHTML = [1,2,4].map(u=>`<button aria-pressed="${p.u===u}" data-u="${u}">${u}</button>`).join("");
  $("seg-ctx").innerHTML = ["16K","64K","128K","256K"].map(c=>{
    const ok = P.some(x=>x.u===p.u&&x.c===c);
    return `<button aria-pressed="${p.c===c}" data-c="${c}" ${ok?"":"disabled"}>${c}</button>`;
  }).join("");
}

function renderHeat(){
  const m = M[heatMetric];
  const columns = P.map((p,i)=>({p,i})).filter(({p}) => overviewWorkload === 'Short' ? p.c === '16K' : p.c !== '16K' || p.u === 4);
  const overviewRows = p => matchingRows(p).filter(r => r[9] === (overviewWorkload === 'Short' ? 'Unified' : 'Separate'));
  $("seg-heat").innerHTML = M.slice(0,3).map((x,i)=>`<button aria-pressed="${i===heatMetric}" data-h="${i}">${x.name}</button>`).join("");
  const variants = [...new Map(columns.flatMap(({p}) => overviewRows(p)).map(r => [[r[0],r[1],r[2],r[9]].join('|'),r])).values()].sort(profileOrder);
  let h = `<tr><th>Model</th><th>Backend</th><th>K/V cache</th>${columns.map(({p})=>`<th scope="col">${p.c}<span class="u">${p.u} user${p.u>1?'s':''}</span></th>`).join('')}</tr>`;
  h += variants.map(variant => `<tr><th scope="row" class="m">${variant[0]}</th><td>${variant[1]}</td><td>${variant[2]}</td>${columns.map(({p,i})=>{
    const r = overviewRows(p).find(x => x[0]===variant[0] && x[1]===variant[1] && x[2]===variant[2]);
    const dl = r ? rowDelta(m, r) : {txt:variant[0].includes('E4B') && p.c === '256K' ? 'Model limit' : 'Outside scope',cls:'neutral',v:null};
    const bg = dl.cls === 'neutral' ? 'transparent' : `color-mix(in srgb, var(--${dl.cls}) ${dl.v === null ? 18 : Math.min(55,10+Math.abs(dl.v)*.5)}%, transparent)`;
    return `<td class="${dl.cls} ${i===sel?'sel':''}" style="background:${bg}">${r ? `<button data-p="${i}">${dl.txt}</button>` : dl.txt}</td>`;
  }).join('')}</tr>`).join('');
  $("heat").innerHTML = variants.length ? h : `<tr><td>Outside scope for these filters</td></tr>`;
}

function keepSelectionInScope(preferred) {
  const current = {u:P[sel].u, c:P[sel].c, workload:overviewWorkload, backend:selectedBackend, cache:selectedCache};
  const allowed = (p, w, b, k) => models.some(model => backends.some(backend =>
    (b === 'All' || b === backend) && caches.some(cache =>
      (includeQ8 || cache !== 'Q8/Q8') && (k === 'All' || k === cache)
      && plannedProfile(p, model, backend, cache, w))));
  if (allowed(P[sel], current.workload, current.backend, current.cache)) return;
  const weights = {u:1000, c:100, workload:20, backend:4, cache:2};
  let best = null;
  for (const [index,p] of P.entries()) for (const workload of ['Short','Long'])
    for (const backend of ['All',...backends]) for (const cache of ['All',...caches]) {
      if ((!includeQ8 && cache === 'Q8/Q8') || !allowed(p,workload,backend,cache)) continue;
      const candidate = {u:p.u,c:p.c,workload,backend,cache};
      const cost = Object.keys(weights).reduce((sum,key) => sum + (candidate[key] === current[key] ? 0 : weights[key] + (key === preferred ? 10000 : 0)),0);
      if (!best || cost < best.cost) best = {index,workload,backend,cache,cost};
    }
  if (best) {
    sel=best.index; overviewWorkload=best.workload; selectedBackend=best.backend; selectedCache=best.cache;
  }
}
function render(preferred = '') {
  keepSelectionInScope(preferred);
  renderPicker(); renderCards(); renderHeat(); renderCapacity(); renderMemory();
}

$("mtp").addEventListener("click", e=>{
  const t = e.target.closest("button"); if(!t || t.disabled) return;
  const attribute = Object.keys(t.dataset)[0];
  const focusSelector = `button[data-${attribute}="${t.dataset[attribute]}"]`;
  const restoreFocus = () => document.querySelector(focusSelector)?.focus({preventScroll:true});
  const cur = P[sel];
  if(t.dataset.workload){ overviewWorkload=t.dataset.workload; render('workload'); restoreFocus(); return; }
  if(t.dataset.backend){ selectedBackend=t.dataset.backend; render('backend'); restoreFocus(); return; }
  if(t.dataset.cache){ selectedCache=t.dataset.cache; render('cache'); restoreFocus(); return; }
  if(t.dataset.memorybackend){ memoryBackend=t.dataset.memorybackend; renderMemory(); restoreFocus(); return; }
  if(t.dataset.capacity){ capacityUsers = t.dataset.capacity === 'All' ? 'All' : +t.dataset.capacity; renderCapacity(); restoreFocus(); return; }
  if(t.dataset.mtp){ capacityMtp = t.dataset.mtp; renderCapacity(); restoreFocus(); return; }
  if(t.dataset.cbackend){ capacityBackend = t.dataset.cbackend; renderCapacity(); restoreFocus(); return; }
  if(t.dataset.memoryusers){ memoryUsers = t.dataset.memoryusers === 'All' ? 'All' : +t.dataset.memoryusers; renderMemory(); restoreFocus(); return; }
  if(t.dataset.memorymtp){ memoryMtp = t.dataset.memorymtp; renderMemory(); restoreFocus(); return; }
  if(t.dataset.memory){ memoryBudget = t.dataset.memory; renderMemory(); restoreFocus(); return; }
  if(t.dataset.u){ const u=+t.dataset.u; sel = P.findIndex(x=>x.u===u&&x.c===cur.c); if(sel<0) sel = P.findIndex(x=>x.u===u); }
  else if(t.dataset.c){ sel = P.findIndex(x=>x.u===cur.u&&x.c===t.dataset.c); }
  else if(t.dataset.h){ heatMetric = +t.dataset.h; }
  else if(t.dataset.p){ sel = +t.dataset.p; }
  render(attribute);
  restoreFocus();
});

function rowDelta(m, row) {
  const result = delta(m, row[m.i]);
  const recorded = row[8]?.[m.i];
  if (recorded && num(row[m.i][0]) && num(row[m.i][1])) result.txt = recorded;
  return result;
}

let capacityUsers = 1, capacityMtp = "off", capacityBackend = "HIP";
let memoryBudget = "small", memoryUsers = 1, memoryMtp = "off", memoryBackend = "All";
function chips(values) {
  return '<div class="chips">' + values.map(v => `<span class="chip">${v}</span>`).join('') + '</div>';
}
function capacityRows(users, mtp, backend) {
  const rows = DATA.contexts[users].filter(r => r[4] === mtp && r[2].split(' / ').includes(backend)).map(r => [...r]);
  for (const block of Object.values(window.BENCHMARK_DATA.views.mtp)) {
    if (block.kind !== 'rows') continue;
    for (const entry of block.rows) {
      const c = entry.cells.map(cell => cell.value);
      if (c.length !== 10 || !['hip','vulkan'].includes(c[1].toLowerCase()) || !c[5].includes(' / ') || !c[6].includes(' / ')) continue;
      const number = v => Number(v.split(' / ')[0].replaceAll(',',''));
      if (number(c[6])/number(c[5]) !== users || c[1].toLowerCase() !== backend.toLowerCase() || c[4] !== mtp) continue;
      const name = modelNames[benchmarkModelId(c[0])] || c[0];
      if (!models.includes(name)) continue;
      const cache = cacheName(c[2].split(' (')[0]), pool = c[2].includes('separate') ? 'Separate' : 'Unified';
      const existing = rows.find(r => r[0]===name && r[3]===cache && r[6]===pool);
      const usable = c[9].startsWith('Pass') ? c[5].split(' / ')[1] : null;
      if (existing) {
        if (usable !== null && (existing[5] === null || Number(usable.replaceAll(',','')) > Number(existing[5].replaceAll(',','')))) { existing[5] = usable; existing[7] = 'Pass'; }
        continue;
      }
      rows.push([name,name.includes('E4B')?'131,068':'262,140',backend,cache,mtp,usable,pool,c[9]]);
    }
  }
  for (const group of cachedRunGroups) {
    for (const c of window.BENCHMARK_DATA.cached?.[group] || []) {
      if (c.status !== 'passed' || c.slots !== users || c.backend.toLowerCase() !== backend.toLowerCase()
          || c.mtp !== (mtp === 'on') || !(c.gpu_spare_mib >= 512)) continue;
      const name = modelNames[benchmarkModelId(c.model)];
      if (!name) continue;
      const usable = (c.context_per_user - (c.mtp ? 4 : 0)).toLocaleString('en-US');
      rows.push([name,name.includes('E4B')?'131,068':'262,140',backend,cacheName(c.cache+'/'+c.cache),mtp,usable,'Separate','Pass']);
    }
  }
  return models.flatMap(model => {
    const candidates = rows.filter(r => r[0] === model && r[5] !== null && (includeQ8 || r[3] !== 'Q8/Q8'));
    candidates.sort((a,b) => Number(b[5].replaceAll(',','')) - Number(a[5].replaceAll(',','')) || caches.indexOf(a[3]) - caches.indexOf(b[3]));
    return candidates.length ? [candidates[0]] : [];
  });
}
function renderCapacity() {
  $("seg-capacity").innerHTML = ["All",1,2,4].map(u => `<button aria-pressed="${u === capacityUsers}" data-capacity="${u}">${u}</button>`).join('');
  $("seg-capacity-mtp").innerHTML = ["All","off","on"].map(state => `<button aria-pressed="${state === capacityMtp}" data-mtp="${state}">${state === "All" ? "All" : state === "off" ? "Off" : "On"}</button>`).join('');
  $("seg-capacity-backend").innerHTML = ["All", "HIP", "Vulkan"].map(backend => `<button aria-pressed="${backend === capacityBackend}" data-cbackend="${backend}">${backend}</button>`).join('');
  const users = capacityUsers === 'All' ? [1,2,4] : [capacityUsers];
  const states = capacityMtp === 'All' ? ['off','on'] : [capacityMtp];
  const backends = capacityBackend === 'All' ? ['HIP','Vulkan'] : [capacityBackend];
  const rows = users.flatMap(u => states.flatMap(mtp => backends.flatMap(backend =>
    capacityRows(u, mtp, backend).map(r => ({row:r, users:u, backend})))));
  const headers = ['Model', 'Model limit · MTP on', 'Users', 'Backend', 'KV cache', 'MTP', 'Usable tokens / user', 'Total --ctx-size', 'KV pool'];
  $("context-table").innerHTML = `<thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>` + (rows.length ? rows.map(({row:r, users, backend}) => {
    const usable = r[5] === null ? null : Number(r[5].replaceAll(',', ''));
    const total = usable === null ? (r[7] || 'Not established') : ((usable + (r[4] === 'on' ? 4 : 0)) * users).toLocaleString('en-US');
    return `<tr><th scope="row" class="m">${r[0]}</th><td>${r[1]}</td><td>${users}</td><td>${backend}</td><td>${r[3]}</td><td>${r[4]}</td><td>${r[5] ?? r[7] ?? 'Not established'}</td><td>${total}</td><td>${r[6]}</td></tr>`;
  }).join('') : `<tr><td colspan="9">No verified context profile for this selection.</td></tr>`) + '</tbody>';
}
function renderMemory() {
  segments('seg-memory-backend',['All',...backends],memoryBackend,'memorybackend');
  const budgets = [['All','All'],['small','≪8 GiB'],['eight','8 GiB'],['sixteen','16 GiB']];
  $("seg-memory").innerHTML = budgets.map(([key,label]) => `<button aria-pressed="${key === memoryBudget}" data-memory="${key}">${label}</button>`).join('');
  $("seg-memory-users").innerHTML = ['All',1,2,4].map(u => `<button aria-pressed="${u === memoryUsers}" data-memoryusers="${u}">${u}</button>`).join('');
  $("seg-memory-mtp").innerHTML = ['All','off','on'].map(state => `<button aria-pressed="${state === memoryMtp}" data-memorymtp="${state}">${state === 'All' ? 'All' : state === 'off' ? 'Off' : 'On'}</button>`).join('');
  const ranges = {All: 'up to 16 GiB', small: 'below 7 GiB', eight: '7–8 GiB', sixteen: 'over 8–16 GiB'};
  $('memory-note').textContent = `Measured contexts · GPU memory ${ranges[memoryBudget]}`;
  const states = memoryMtp === 'All' ? ['off','on'] : [memoryMtp];
  const budget = memoryBudget === 'sixteen' || memoryBudget === 'All' ? 16 : 8;
  const measured = new Map();
  const addMemory = row => {
    if (!models.includes(row[0]) || !backends.includes(row[1]) || !Number.isFinite(row[6]) || row[6] > 16
        || (memoryUsers !== 'All' && row[3] !== memoryUsers) || !states.includes(row[5])
        || (!includeQ8 && row[2] === 'Q8/Q8') || (memoryBackend !== 'All' && row[1] !== memoryBackend)) return;
    const key = JSON.stringify([...row.slice(0,6),row[7]]);
    const previous = measured.get(key);
    if (!previous || row[6] > previous[6]) measured.set(key,row);
  };
  for (const block of Object.values(window.BENCHMARK_DATA.views.mtp)) {
    if (block.kind !== 'rows') continue;
    for (const entry of block.rows) {
      const c = entry.cells.map(cell => cell.value);
      if (c.length !== 10 || !c[5].includes(' / ') || !c[6].includes(' / ')
          || !c[9].startsWith('Pass') || !/^\d+(?:\.\d+)?$/.test(c[7])) continue;
      const allocated = Number(c[5].split(' / ')[0].replaceAll(',',''));
      const users = Number(c[6].split(' / ')[0].replaceAll(',','')) / allocated;
      const backend = c[1].toLowerCase() === 'hip' ? 'HIP' : c[1].toLowerCase() === 'vulkan' ? 'Vulkan' : c[1];
      addMemory([modelNames[benchmarkModelId(c[0])],backend,cacheName(c[2].split(' (')[0]),users,
        allocated / 1024 + 'K',c[4],Number(c[7]),c[2].includes('separate') ? 'Separate' : 'Unified']);
    }
  }
  for (const p of P.filter(p => memoryUsers === 'All' || p.u === memoryUsers)) {
    for (const state of states) {
      const stateIndex = state === 'off' ? 0 : 1;
      for (const r of p.rows) {
        const peak = r[7][stateIndex];
        if ((!includeQ8 && r[2] === 'Q8/Q8') || (memoryBackend !== 'All' && r[1] !== memoryBackend)
            || !num(peak) || peak > 16) continue;
        addMemory([r[0],r[1],r[2],p.u,p.c,state,peak,r[9]]);
      }
    }
  }
  const rows = [...measured.values()].filter(r => {
    const peak = Number(r[6]);
    return memoryBudget === 'All' || (memoryBudget === 'small' ? peak < 7
      : memoryBudget === 'eight' ? peak >= 7 && peak <= 8 : peak > 8);
  }).sort((a,b) => profileOrder(a,b) || a[3]-b[3] || parseInt(a[4])-parseInt(b[4]) || a[5].localeCompare(b[5]) || a[7].localeCompare(b[7]));
  $("memory-cards").innerHTML = rows.length ? rows.map(r => `<article class="card">
    <div class="card-head"><div><h3 class="card-title">${r[0]}</h3>${chips([r[3]+' user'+(r[3]>1?'s':''),r[4]+'/user','MTP '+r[5],r[1],'KV '+r[2],r[7]+' KV'])}</div></div>
    <div class="card-value">${r[6].toFixed(2)} <small>GiB peak GPU</small></div>
    <div class="memory-meter" aria-hidden="true"><span style="width:${Math.min(100,Number(r[6])/budget*100)}%"></span></div>
  </article>`).join('') : '<p class="small">No measured configuration fits this selection.</p>';
}
$('include-q8').addEventListener('click', e => {
  const button = e.target.closest('button[data-q8]');
  if (!button) return;
  includeQ8 = button.dataset.q8 === 'on';
  $('include-q8').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.q8 === 'on') === includeQ8)));
  if (!includeQ8 && selectedCache === 'Q8/Q8') selectedCache='All';
  render();
});
render();

function timingDetails(profile, row) {
  if (row[9] !== 'Unified' || profile.c !== '16K') return '';
  const records = Object.values(window.BENCHMARK_DATA.views.mtp).filter(b => b.kind === 'rows').flatMap(b => b.rows)
    .map(r => r.cells.map(c => c.value)).filter(c => c.length === 16 && modelNames[benchmarkModelId(c[0])] === row[0]
      && c[1].toLowerCase() === row[1].toLowerCase() && Number(c[2]) === profile.u && cacheName(c[4]) === row[2]);
  if (!records.length) return '';
  return '<details><summary>Timing details</summary><div class="heat-wrap"><table class="heat"><thead><tr><th>MTP</th><th>Draft accepted %</th><th>Request s</th><th>End-to-end t/s ± SD</th></tr></thead><tbody>'
    + records.map(c => `<tr><td>${c[3]}</td><td>${c[12]}</td><td>${c[11]}</td><td>${c[8]}</td></tr>`).join('')
    + '</tbody></table></div><p class="small">End-to-end throughput includes prefill; Speed above excludes it.</p></details>';
}
const savedAnswers = Object.values(window.BENCHMARK_DATA.views.mtp).filter(b => b.kind === 'answer')
  .flatMap(block => block.answers).map(answer => ({...answer,
    search:[answer.label,answer.text,...answer.paragraphs].join(' ').toLowerCase()}));
function renderAnswers() {
  const query = $('answer-filter').value.toLowerCase().trim();
  const selected = savedAnswers.filter(a => !query || a.search.includes(query));
  $('answer-select').replaceChildren(...selected.map((answer,i) => new Option(`${i+1}. ${answer.label}`,String(i))));
  const show = () => {
    const answer = selected[Number($('answer-select').value)];
    const target = $('answer-view');
    target.replaceChildren();
    if (answer) {
      const picture = document.createElement('img');
      picture.src = answer.src;
      picture.alt = answer.image;
      const prompt = document.createElement('p');
      prompt.textContent = answer.paragraphs[0] || '';
      const text = document.createElement('pre');
      text.textContent = answer.text;
      target.append(picture,prompt,text);
      for (const value of answer.paragraphs.slice(1)) {
        const paragraph = document.createElement('p');
        paragraph.textContent = value;
        target.append(paragraph);
      }
    } else target.textContent = 'No matching saved answers';
    const image = answer?.image;
    const reference = window.BENCHMARK_DATA.models.reference.answers[image]?.text;
    $('image-reference').hidden = !reference;
    $('image-reference').querySelector('p').textContent = reference || '';
  };
  $('answer-select').onchange=show;
  show();
}
$('answer-filter').addEventListener('input',renderAnswers);
renderAnswers();
