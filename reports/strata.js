'use strict';
const strata = window.BENCHMARK_DATA.catalog.strata;
const put = (id, value) => { document.getElementById(id).textContent = value; };
const evaluation = strata.evaluation_config;
const verified = strata.results.filter(r => r.status === 'verified');
put('hardware', `${strata.model} · ${strata.hardware}`);
put('settings', `HIP · ${evaluation.context.toLocaleString()} context limit · one active request · thinking enabled · speculation ${strata.settings.speculation} · ${strata.settings.vram_reserve_mib} MiB VRAM reserve`);
function showRequest(result) {
  const request = result.request;
  put('output-budget', `Output limit: ${request.max_tokens.toLocaleString()} tokens total, including up to ${request.reasoning_budget_tokens.toLocaleString()} reasoning tokens.`);
  put('evaluation-call', [
    'curl -N \\',
    '  http://127.0.0.1:8080/v1/chat/completions \\',
    "  -H 'Content-Type: application/json' \\",
    "  --data-binary @- <<'JSON'",
    JSON.stringify(request, null, 2),
    'JSON'
  ].join('\n'));
}
function addMeasurement(r) {
  const tr = document.getElementById('results').insertRow();
  const g = r.generation;
  for (const value of [r.quant, r.kv || 'int8', r.placement, g.ttft_seconds.toFixed(2), g.first_content_seconds.toFixed(2),
    g.wall_seconds.toFixed(2), g.tokens_per_second.toFixed(1), (r.peak_ram_bytes / 2 ** 30).toFixed(2),
    r.peak_gpu_bytes == null ? 'Not recorded' : (r.peak_gpu_bytes / 2 ** 30).toFixed(2)]) {
    tr.insertCell().textContent = value;
  }
}
function addEvaluation(quant, kv, generation, issue, fix) {
  const row = document.getElementById('exporter-repairs').insertRow();
  for (const value of [quant, kv, generation.tokens_per_second.toFixed(1), issue, fix]) {
    row.insertCell().textContent = value;
  }
}
if (verified.length) {
  document.getElementById('verified-results').hidden = false;
  const select = document.getElementById('request-model');
  select.hidden = false;
  document.getElementById('request-label').hidden = false;
  verified.forEach(r => {
    if (r.quant === 'IQ2_XS') put('iq2-decode', r.generation.tokens_per_second.toFixed(1));
    const option = document.createElement('option');
    option.value = option.textContent = r.quant;
    select.append(option);
    addMeasurement(r);
    if (r.quant === strata.iq3_coding_kv.quant) addMeasurement(strata.iq3_coding_kv);
    addEvaluation(r.quant, 'INT8', r.generation, r.exporter_issue, r.exporter_fix);
    if (r.quant === strata.iq3_coding_kv.quant) {
      const q4 = strata.iq3_coding_kv;
      addEvaluation(q4.quant, q4.kv, q4.generation, q4.original_issue, q4.verified_fix);
    }
  });
  put('validation-note', [verified.every(r => r.program_status === 'repaired') ? 'The short-prompt programs passed after repair.' : '', strata.validation_note].filter(Boolean).join(' '));
  select.value = verified.find(r => r.quant === 'IQ3_S')?.quant || verified[0].quant;
  showRequest(verified.find(r => r.quant === select.value));
  select.addEventListener('change', () => showRequest(verified.find(r => r.quant === select.value)));
}

const coding = strata.iq3_long_coding;
put('long-coding-method', `Both caches received the same ${coding.input_tokens.toLocaleString()}-token history of code changes, GPU readings and requirements. Each had one 256K slot, a 2 GiB conversation cache and a 16K output budget. We ran each original program once.`);
for (const run of coding.results) {
  const item = document.createElement('li');
  item.textContent = `${run.kv === 'int8' ? 'INT8' : run.kv} · ${run.verdict}. ${run.finding.replace(/\.$/, '')}`;
  document.getElementById('long-coding-results').append(item);
}

const single = strata.iq3_cache_limit;
put('conversation-method', `IQ3_S served one request at a time with ${single.context.toLocaleString()} context tokens, a ${single.output_cap.toLocaleString()}-token output cap and ${single.high_gib}G/${single.max_gib}G RAM limits. Each return followed a short side chat.`);
const cacheAttempts = new Map(single.attempts.map(run => [run.source, run]));
for (const key of single.display) {
  const run = cacheAttempts.get(key);
  const result = run.returns.at(-1);
  const tr = document.getElementById('conversation-results').insertRow();
  for (const value of [`${run.budget_mib / 1024} GiB`, run.returns.length === 1 ? 'First' : 'Second',
    result.prompt.toLocaleString(), result.reused.toLocaleString(), result.ttft.toFixed(2),
    (run.peak_ram_bytes / 2 ** 30).toFixed(2)]) tr.insertCell().textContent = value;
}
const boundary = cacheAttempts.get(single.reliable[0]);
const resumed = boundary.returns.at(-1);
const missed = cacheAttempts.get(single.larger.at(-1)).returns.at(-1);
put('conversation-limit', `Use a ${boundary.budget_mib / 1024} GiB conversation cache for repeated switching. It restored ${resumed.reused.toLocaleString()} tokens on consecutive returns in two runs. At ${missed.prompt.toLocaleString()} tokens, both 2 and 4 GiB budgets required a full reread to stay within the RAM limit.`);

for (const row of strata.iq3_alternating_chats.rows) {
  const tr = document.getElementById('ab-results').insertRow();
  const {A, B} = row.chats;
  for (const value of [`${row.cache_mib / 1024} GiB`, `${row.high_gib}/${row.max_gib}`,
    A.cached.toLocaleString(), A.ttft.toFixed(2), B.cached.toLocaleString(), B.ttft.toFixed(2),
    (row.peak_ram_bytes / 2 ** 30).toFixed(2)]) tr.insertCell().textContent = value;
}

const slotRuns = strata.iq3_two_slots.runs;
const solo = slotRuns.find(run => run.users.length === 1);
const pairs = [false, true].map(mtp => slotRuns.find(run => run.users.length === 2 && run.batch_mtp === mtp));
function comparisonRow(label, values) {
  const tr = document.getElementById('two-slot-results').insertRow();
  const th = document.createElement('th');
  th.scope = 'row';
  th.textContent = label;
  tr.append(th);
  for (const value of values) tr.insertCell().textContent = value;
}
for (const {user} of pairs[0].users) {
  const results = pairs.map(run => run.users.find(result => result.user === user));
  comparisonRow(`User ${user} · TTFT s`, results.map(r => r.ttft_seconds.toFixed(2)));
  comparisonRow(`User ${user} · decode t/s`, results.map(r => r.decode_tps.toFixed(1)));
}
comparisonRow('Total output t/s', pairs.map(r => r.output_tps_including_prefill.toFixed(2)));
comparisonRow('Stream overlap s', pairs.map(r => r.overlap_seconds.toFixed(2)));
const [userA] = solo.users;
put('two-slot-throughput', `User A alone on the same server used solo MTP: ${userA.ttft_seconds.toFixed(2)} s TTFT, ${userA.decode_tps.toFixed(1)} decode t/s and ${solo.output_tps_including_prefill.toFixed(2)} total output t/s.`);

const streamed = strata.iq3_two_slots.streamed_kv_capacity;
put('streamed-kv-capacity', `With a ${(streamed.kv_resident/1024)}K resident KV window, both GPU slot buffers fit in ${streamed.gpu_slot_buffers_gib.toFixed(2)} GiB. Startup then exhausted the ${(streamed.service_memory_max_bytes / 2 ** 30).toFixed(2)} GiB service RAM allowance before either user could generate.`);

const cold = strata.iq3_two_slots.cold_capacity;
const [coldA, coldB] = cold.requests;
put('two-slot-capacity', `Near the 128K limit, both users sent ${coldA.input_tokens.toLocaleString()} uncached tokens together and each received ${coldA.output_tokens} tokens. First tokens arrived after ${coldA.ttft.toFixed(0)} s and ${coldB.ttft.toFixed(0)} s. The server read the two fresh histories in turn.`);

const [resumedA, resumedB] = strata.iq3_two_slots.displaced_histories.requests;
put('two-slot-resume', `After other chats used both slots, B reused ${resumedB.reused.toLocaleString()} tokens and received its first token in ${resumedB.ttft_seconds.toFixed(2)} s. A reread ${resumedA.prompt.toLocaleString()} tokens and waited ${resumedA.ttft_seconds.toFixed(2)} s. Strata dropped A’s saved history to stay within its RAM limit.`);

const baselinePair = pairs[0];
const contextRows = [[strata.iq3_two_slots.context_per_user, 'int8',
  ...baselinePair.users.flatMap(u => [u.ttft_seconds, u.decode_tps]), baselinePair.output_tps_including_prefill]];
for (const r of strata.iq3_two_slots.larger_contexts) {
  contextRows.push([r.context, r.kv, ...r.users.flatMap(u => [u.ttft, u.decode]), r.total_tps]);
}
for (const [ctx, kv, aWait, aDecode, bWait, bDecode, total] of contextRows) {
  const tr = document.getElementById('larger-contexts').insertRow();
  for (const value of [`${ctx / 1024}K`, kv, aWait.toFixed(2), aDecode.toFixed(1), bWait.toFixed(2),
    bDecode.toFixed(1), total.toFixed(2)]) tr.insertCell().textContent = value;
}

const largestCold = strata.iq3_two_slots.largest_cold_capacity;
const coldWait = largestCold.rows.map(row => row.ttft / 60).sort((a, b) => a - b);
put('largest-cold', `With ${largestCold.kv} KV, two 256K slots accepted ${largestCold.rows[0].prompt.toLocaleString()} cold input tokens per user and each generated ${largestCold.rows[0].output}. First tokens arrived after ${coldWait[0].toFixed(1)} and ${coldWait[1].toFixed(1)} minutes. The answers arrived in turn. Peak service RAM was ${(largestCold.peak_service_ram_bytes / 2**30).toFixed(2)} GiB.`);
