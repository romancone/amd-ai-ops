#!/usr/bin/env python3
"""Render condensed evidence and one answer per image; raw logs remain external."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def format_mtp_html(document):
    """Put blocks and table rows on separate lines; preserve saved answers verbatim."""
    containers = r'html|head|body|main|nav|header|footer|section|article|div|ul|ol|table|thead|tbody|tfoot|details'
    elements = r'h[1-6]|p|li|tr|summary|title|meta|link|img'
    parts = re.split(r'(<pre\b[^>]*>.*?</pre>)', document, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            rf'\s*(?=</?(?:{containers})\b|<(?:{elements})\b)',
            '\n', parts[i]).strip()
    return '\n'.join(part for part in parts if part) + '\n'


def gpu_only_columns(head, values):
    """Keep GPU memory in HTML tables; preserve host readings in source data."""
    keep = [i for i, name in enumerate(head)
            if 'GPU' in name or not re.search(r'\b(?:RAM|GTT)\b', name)]
    combined = {i for i in keep if ' / RAM / GTT' in head[i]}
    return ([head[i].replace(' / RAM / GTT', '') for i in keep],
            [[row[i].split(' / ', 1)[0] if i in combined and isinstance(row[i], str)
              else row[i] for i in keep] for row in values])

def render_mtp(run, repo):
    import html
    import statistics as stats
    esc = lambda value: html.escape(str(value))
    load = lambda path: json.loads(path.read_text())
    fmt = lambda value: f'{value:.2f}' if isinstance(value, (float, int)) else '—'
    reference = load(repo/'evaluation/reference.json')
    reviews = load(run/'reviews-luna.json') if (run/'reviews-luna.json').exists() else {}
    state = load(run/'status-mtp.json')
    headers = ['Model', 'Backend', 'Users', 'MTP', 'K/V cache', 'FA', 'Prompt t/s/user', 'Gen t/s/user',
               'Total t/s ± SD', 'Speed ratio (on/off)', 'TTFT ms', 'Request s', 'Draft accept %',
               'Image quality %', 'Peak GPU GiB', 'Peak RAM GiB', 'Status']
    rows, answers, summaries = [], {}, {}
    for path in sorted((run/'matched').glob('*/summary.json')):
        item = load(path)
        summaries[item['id']] = item
    for key, item in summaries.items():
        requests = [r for speed in item['speeds'] for r in speed.get('wave', {}).get('requests', [])]
        timings = [r.get('response', {}).get('timings', {}) for r in requests]
        mean = lambda values: stats.mean(values) if values else None
        qualities = []
        for wave in item['images']:
            for sample in wave.get('wave', {}).get('requests', []):
                image, text = sample.get('image'), sample.get('text', '')
                if not image or not text:
                    continue
                prompt = reference['answers'][image]['prompt']
                digest = hashlib.sha256((image+'\0'+prompt+'\0'+text).encode()).hexdigest()
                review = reviews.get(digest, {})
                if review:
                    if review.get('answer_sha256') != hashlib.sha256(text.encode()).hexdigest():
                        raise ValueError('Review does not match answer: '+digest)
                    quality = sum(review[k] for k in reference['rubric']['dimensions'])
                    qualities.append(quality)
                else:
                    quality = None
                entry = answers.setdefault(digest, {'configurations': [], 'body': ''})
                entry['configurations'].append(key)
                entry['body'] = (f'<summary>{esc(image)} · {fmt(quality)}%</summary>'
                    f'<img src="../media/qengineering/{esc(image)}.jpg" alt="{esc(image)}">'
                    f'<p>{esc(prompt)}</p><pre>{esc(text)}</pre>'
                    f'<p>Finish: {esc(sample.get("response", {}).get("choices", [{}])[0].get("finish_reason", "unknown"))}</p>'
                    f'<p>{esc(review.get("note", review.get("reason", "Awaiting Luna-6 review")))}</p>')
        trials = [v['trial'] for v in item['speeds']+item['images']]
        peak = lambda name: max([t.get('peak_memory_mib', {}).get(name, 0) for t in trials] or [0])/1024
        off = summaries.get(key.replace('-mtp1', '-mtp0'), {})
        speedup = (item.get('aggregate_tps_mean', 0)/off['aggregate_tps_mean']
                   if item['mtp'] and off.get('aggregate_tps_mean') else None)
        proposed = sum(t.get('draft_n', 0) for t in timings)
        accepted = sum(t.get('draft_n_accepted', 0) for t in timings)
        rows.append([item['model']['id'], item['backend'], item['slots'], 'on' if item['mtp'] else 'off',
            item.get('cache', 'f16')+'/'+item.get('cache', 'f16'), 'on', mean([t['prompt_per_second'] for t in timings if 'prompt_per_second' in t]),
            mean([t['predicted_per_second'] for t in timings if 'predicted_per_second' in t]),
            (fmt(item['aggregate_tps_mean'])+' ± '+fmt(item.get('aggregate_tps_sd'))) if item.get('aggregate_tps_mean') is not None else None, speedup,
            mean([r['ttft_seconds']*1000 for r in requests if r.get('ttft_seconds') is not None]),
            mean([r['seconds'] for r in requests]), 100*accepted/proposed if proposed else None,
            mean(qualities) if len(qualities)==4 else None, peak('vram_mib'), peak('rss_mib'), item['status']])
    supplementary = [row for row in rows if row[2] == 1 and row[3] == 'on']
    rows = [row for row in rows if not (row[2] == 1 and row[3] == 'on')]
    baselines = []
    for path in sorted((run/'baseline').glob('*/bench.json')):
        data = load(path)
        pp = next((v for v in data if v.get('n_prompt')==512), {})
        tg = next((v for v in data if v.get('n_gen')==128), {})
        baselines.append([path.parent.name, 'on', 'f16/f16', fmt(pp.get('avg_ts'))+' ± '+fmt(pp.get('stddev_ts')), fmt(tg.get('avg_ts'))+' ± '+fmt(tg.get('stddev_ts'))])
    cap_headers = ['Model', 'Backend', 'FA', 'Max allocated / usable tokens',
                   'KV MiB', 'Practical allocated / usable tokens', 'Practical GPU free MiB', 'Upper test tokens',
                   'Full-prompt t/s', 'Generation t/s', 'Peak GPU GiB', 'Peak RAM GiB',
                   'Peak GTT GiB', 'Boundary / status']
    caps = []
    cap_passed = 0
    capacity_failures = []
    for path in sorted((run/'capacity').glob('*/summary.json')):
        item = load(path)
        if item['slots'] != 1 or item['mtp'] or item['cache'] != 'f16':
            continue
        limit = item['largest_passing_per_user']
        cap_passed += bool(limit and item['practical_per_user'])
        if not limit:
            lost = [load(q)['context_per_user'] for q in path.parent.glob('ctx-*/full/result.json')
                    if load(q).get('status') == 'device_lost']
            if lost:
                capacity_failures.append(item['model']+'/'+item['backend']+': ErrorDeviceLost at '+
                                         ', '.join(f'{n:,}' for n in sorted(lost))+' allocated tokens. No full-capacity limit is established; its 16K timing and image tests passed.')
        guard = item.get('guard_tokens_per_user', 0)
        pair = lambda value: f'{value:,} / {value-guard:,}' if value else None
        full_dir = item.get('full_result_dir', 'full')
        peaks = {'vram_mib': 0, 'rss_mib': 0, 'gtt_mib': 0}
        kv_mib = None
        full_timing = {}
        if limit:
            for kind in [full_dir, 'image-memory']:
                trial = load(path.parent/f'ctx-{limit}'/kind/'trial.json')
                if kind == full_dir:
                    full_result = load(path.parent/f'ctx-{limit}'/kind/'result.json')
                    full_timing = full_result['wave']['requests'][0]['response']['timings']
                    kv_mib = sum(float(v) for line in trial.get('allocations', [])
                        for v in re.findall(r'(?:ROCm|Vulkan)\d+ KV buffer size\s*=\s*([\d.]+) MiB', line))
                for field in peaks:
                    peaks[field] = max(peaks[field], trial.get('peak_memory_mib', {}).get(field, 0))
        failed_memory = False
        if not limit:
            native = next(m['max_context'] for m in load(run/'manifest.json')['models'] if m['id']==item['model'])
            failed_trial = path.parent/f'ctx-{native}'/full_dir/'trial.json'
            if failed_trial.exists():
                trial = load(failed_trial)
                failed_memory = True
                peaks.update(trial.get('peak_memory_mib', {}))
                kv_mib = sum(float(v) for line in trial.get('allocations', [])
                    for v in re.findall(r'(?:ROCm|Vulkan)\d+ KV buffer size\s*=\s*([\d.]+) MiB', line))
        status = ('model limit' if limit==next(m['max_context'] for m in load(run/'manifest.json')['models'] if m['id']==item['model'])
                  else '4K boundary' if item['boundary_resolved'] else 'lower bound' if limit else 'not established')
        if not limit:
            status += ': '+', '.join(str(p.get('full', p.get('probe'))) for p in item['points'][-1:])
        practical_free = None
        if item['practical_per_user']:
            practical_peak = max(load(path.parent/f"ctx-{item['practical_per_user']}"/kind/'trial.json')
                                 ['peak_memory_mib']['vram_mib'] for kind in [full_dir, 'image-memory'])
            practical_free = item.get('gpu_budget_mib', 16384)-practical_peak
        caps.append([item['model'], item['backend'], 'on', pair(limit), kv_mib,
            pair(item['practical_per_user']), practical_free, item['upper_bound_per_user'],
            full_timing.get('prompt_per_second'), full_timing.get('predicted_per_second'),
            peaks['vram_mib']/1024 if limit or failed_memory else None, peaks['rss_mib']/1024 if limit or failed_memory else None,
            peaks['gtt_mib']/1024 if limit or failed_memory else None, status])
    def table(head, values):
        head, values = gpu_only_columns(head, values)
        body = ''.join('<tr>'+''.join('<td>'+esc(fmt(v) if v is None or isinstance(v,float) else v)+'</td>' for v in row)+'</tr>' for row in values)
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in head)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    doc += '<title>AMD AI Ops — MTP and users</title><link rel="stylesheet" href="style.css"><main class="mtp-report">'
    doc += '<nav class="report-nav" aria-label="Reports"><a href="index.html">Recommendations</a><a href="report.html">Model tests</a><a href="mtp.html" aria-current="page">MTP and users</a></nav><h1>MTP, users and cache capacity</h1>'
    doc += f'<p>RX 6900 XT 16 GiB · llama.cpp <code>{esc(state["source_revision"][:9])}</code>.</p>'
    doc += f'<p>Matched results: {len(rows)}/90. Capacity checks: {len(caps)}/6 recorded, {cap_passed} validated, {len(caps)-cap_passed} unresolved.</p>'
    doc += '<h2>1. Detailed results</h2><h3>1.1 Matched tests at 16K per user</h3><p><strong>Shared settings:</strong></p><ul><li>Profiles: 18 single-user with MTP off; 72 for two/four users with MTP off/on</li><li>K/V cache: f16/f16, Q8/Q8 or Q4/Q4</li><li>Speed tests: three batches, 512 input and 128 output tokens per request</li><li>Image quality: four images, one answer each, scored against Luna-6</li><li>Server: restarted before each image wave and timing batch</li></ul>'
    doc += table(headers, rows)
    if supplementary:
        doc += '<h3>1.2 Single-user MTP-on results</h3><details><summary>Show six f16/f16 profiles</summary>'+table(headers, supplementary)+'</details>'
    doc += '<h3>1.3 Ordinary llama-bench</h3><p><strong>Settings:</strong></p><ul><li>Users: 1</li><li>Context: 16K</li><li>MTP: off</li><li>Microbatch: 512</li><li>Speed tests: pp512 and tg128, three runs each</li></ul>'
    doc += table(['Model / backend', 'FA', 'Cache', 'pp512 ± SD', 'tg128 ± SD'], baselines)
    doc += ('<h3>1.4 Full-context measurements</h3><p id="context-limits">Six capacity checks.</p>'
            '<p><strong>Settings:</strong></p><ul><li>Users: 1</li><li>MTP: off</li><li>K/V cache: f16/f16</li></ul>'
            '<p>Context values are exact tokens. Memory refers to the largest passing context, or the failed native attempt when no pass exists. '
            'Practical GPU free is the smallest sampled headroom during full-fill and image checks. Capacity does not establish long-context image quality.</p>')
    boundary_path = run/'strict-boundary-checks.json'
    boundary_notice = ''
    if boundary_path.exists():
        boundary = load(boundary_path)
        if boundary:
            boundary_notice = 'Earlier unguarded MTP boundary checks (outside the six checks): '+ '; '.join(
                f"{v['model']}/{v['backend']}, {v['slots']} user, {v['cache']}, {v['context_per_user']:,} tokens: generated {', '.join(map(str, v['outputs']))} of 128 requested" for v in boundary)+'. These are not memory failures or capacity passes.'
            doc += '<p>'+esc(boundary_notice)+'</p>'
    doc += ''.join('<p>'+esc(note)+'</p>' for note in capacity_failures)
    doc += table(cap_headers, caps)
    doc += '<h3>1.5 Image answers</h3><p>Identical image/prompt/answer combinations share one view and judgement.</p>'+''.join(
        '<details>'+value['body']+'<p>Configurations: '+esc(', '.join(value['configurations']))+'</p></details>'
        for value in answers.values())
    doc += '<footer class="doc-links"><a href="../README.md">Repository guide</a><a href="../MODELS.md">Model settings</a><a href="../llama.md">Installation / Runme</a><a href="../BENCHMARK.md">Benchmark tables</a><a href="../evaluation/README.md">How tests are scored</a></footer></main></html>'
    (repo/'reports/mtp.html').write_text(format_mtp_html(doc))
    def markdown(head, values):
        return '\n'.join(['| '+' | '.join(head)+' |', '| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(str(fmt(v) if v is None or isinstance(v,float) else v).replace('|','/') for v in row)+' |' for row in values])
    save_path = run/'condensed-tables.md'
    save_path.write_text('## MTP at 16K per user\n\n'+markdown(headers,rows)+'\n\n## Cache capacity\n\n'+markdown(cap_headers,caps)+'\n')
    benchmark = repo/'BENCHMARK.md'
    prefix = benchmark.read_text().split('## 5. MTP and simultaneous users')[0].rstrip()
    intro = f'''

## 5. MTP and simultaneous users

[Full MTP report and image answers](reports/mtp.html). {len(rows)}/90 configurations recorded; {sum(v["status"] == "passed" and not (v["slots"] == 1 and v["mtp"]) for v in summaries.values())} passed.
16K per user means 16K/32K/64K total. Cache: f16/f16, q8_0/q8_0 and q4_0/q4_0; FA and unified KV on;
BF16 projectors; thinking off; at most three draft tokens. Each row has three
simultaneous timing batches (512 input / 128 output tokens per request) and
four image answers once each. Servers restart
between batches. Qwen uses the dedicated MTP-GGUF target in both modes. The timing prompt is a
fixed inventory/checklist task, not a broad workload sample.

Prompt and generation rates are per-user means from server timings. Total t/s includes prompt time;
speed ratio divides MTP-on total t/s by MTP-off total t/s for the same profile. TTFT means time to first token. Image quality
averages four Luna-6 scores. Small differences do not establish general accuracy.
Each slot allocates 16,384 tokens; the timing/image requests do not fill it.
MTP capacity reserves four tokens per user; full-fill results are in section 6. GPU and process RAM are sampled peaks.
Ordinary pp512/tg128 stays separate.

'''
    capacity_intro = f'''

## 6. Cache capacity

{len(caps)}/6 checks recorded: {cap_passed} validated, {len(caps)-cap_passed} unresolved. One user, MTP off, f16/f16. Status: {state['status']}. Context values are exact tokens.
Allocated is the KV pool per user; usable is input plus output.
These six checks use MTP off, so allocated and usable context are equal.
Input = allocated minus 128; output = 128. No MTP capacity claim is made.
Memory refers to the largest passing context, or the failed native attempt when no pass exists. Practical GPU free is the
smallest sampled headroom during the practical full fill and image check;
it must be at least 512 MiB. Memory is sampled once per second.
Lower bounds are not proven maxima. Upper bounds may be tested failures or
conservative memory exclusions. Capacity does not establish long-context image quality.
KV MiB sums logged GPU KV buffers; recurrent state is separate.
GTT includes host staging buffers and is not, by itself, proof of GPU memory spill.
RAM and GPU-accessible host memory (GTT) can overlap; do not add them together.

'''
    baseline_intro = '\n\n### Ordinary llama-bench at 16K\n\nThree pp512/tg128 repetitions, MTP off, one user, microbatch 512.\nThese are separate from server MTP timings.\n\n'
    supplement = '\n\n### Earlier single-user MTP-on results\n\nSaved f16/f16 checks, outside the approved 90-configuration matrix.\n\n'+markdown(headers,supplementary) if supplementary else ''
    benchmark.write_text(prefix+intro+markdown(headers,rows)+supplement+baseline_intro+
        markdown(['Model / backend', 'FA', 'Cache', 'pp512 ± SD', 'tg128 ± SD'], baselines)+
        capacity_intro+(boundary_notice+'\n\n' if boundary_notice else '')+'\n\n'.join(capacity_failures)+'\n\n'+markdown(cap_headers,caps)+'\n')
    print(f'Rendered {len(rows)} matched rows, {len(caps)} capacity curves and {len(answers)} unique answers.')


def capacity_failure_memory(run, data):
    """Read recorded GPU usage or minimum allocation requirements for failures."""
    load = lambda path: json.loads(path.read_text())
    labels = {'gemma12-Q4_0': 'Gemma 12B Q4_0', 'gemma-e4b-Q4_0': 'Gemma E4B Q4_0',
              'qwen35-MTP-Q4_K_M': 'Qwen 9B MTP Q4_K_M'}
    cases = load(run/'status.json')['cases'] if (run/'status.json').exists() else []
    readings = {}
    for row in data['failures']:
        model, backend, users, context, cache, mtp, outcome = row
        key = tuple(row[:6])
        case = next((c for c in cases if (labels.get(c['model']), c['backend'].upper(), c['slots'])
                     == (model, backend, users) and c['mtp'] == (mtp == 'on')), None)
        attempt = next((a for a in case['attempts']
                        if a['cache'] == cache.split('/')[0] and
                        a.get('allocated_context_per_user', a.get('context')) == context), {}) if case else {}
        if 'floor_mib' in attempt:
            readings[key] = (f"≥{attempt['floor_mib']/1024:.2f} calculated", '—')
            continue
        source = Path(attempt.get('result', attempt.get('reused', '/missing')))
        if not source.exists() and str(source).startswith('/var/tmp/'):
            relative = Path(*source.parts[4:])
            candidates = [run/relative, *run.parent.glob(f'*/node2/{relative}')]
            source = next((p for p in candidates if p.exists()), source)
        trial = load(source).get('trial', {}) if source.exists() else {}
        if not trial and case:
            name = f"{case['model']}-{case['backend']}-n{users}-mtp{int(mtp == 'on')}-{cache.split('/')[0]}"
            saved = list(run.parent.glob(f'*/node2/capacity/{name}/ctx-{context}/full-guard-0/trial.json'))
            if len(saved) == 1:
                trial = load(saved[0])
        if outcome == 'Fail: oom':
            log = source.with_name('server.log')
            content = log.read_text() if log.exists() else ''
            failed = re.search(r'allocating ([\d.]+) MiB on device 0:.*out of memory', content)
            if failed:
                buffers = re.findall(r'ROCm0 (?:model|KV|RS|compute) buffer size =\s*([\d.]+) MiB', content[:failed.start()])
                required = sum(map(float, buffers)) + float(failed[1])
                readings[key] = (f'≥{required/1024:.2f} calculated', '—')
            continue
        peak = trial.get('peak_memory_mib', {}).get('vram_mib')
        guard = source.with_name('headroom-failure.json')
        if guard.exists():
            peak = max(peak or 0, load(guard).get('sample', {}).get('vram_mib', 0))
        spare = re.search(r'([\d]+) MiB (?:peak )?GPU spare|([\d]+) MiB peak spare', outcome)
        free = next(v for v in spare.groups() if v) if spare else '—'
        readings[key] = (f'{peak/1024:.2f}' if peak is not None else '—', free)
    return readings


def render_capacity_summary(run, repo, html_only=False):
    """Publish the approved pass/fail cases without changing speed or image results."""
    import html
    data = json.loads((run/'capacity-summary.json').read_text())
    esc = lambda value: html.escape(str(value))
    headers, rows = data['headers'], data['rows']
    def table(head, values):
        head, values = gpu_only_columns(head, values)
        return '<div class="table-wrap"><table><thead><tr>'+''.join(
            '<th>'+esc(v)+'</th>' for v in head)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table></div>'
    def markdown(head, values):
        return '\n'.join(['| '+' | '.join(head)+' |', '| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in values])
    intro = ('We fill each user’s context to check GPU memory. Smaller contexts with the same settings '
             'should fit with more room. These are verified limits, not exact maximums.')
    summary = data['summary'].replace('capacity cases passed at the settings below.',
                                     'cases have a passing configuration.')
    notes = data['conclusions']
    failures = data['failures']
    failure_headers = data['failure_headers']
    other_headers, other_rows = data['other_headers'], data['other_rows']
    other_intro = ('Practical limits leave at least 512 MiB GPU spare. '
                   'Memory includes image checks where recorded; '
                   'text capacity does not establish long-context image quality.')
    details = ('<details><summary>Other measured contexts</summary><p>'+esc(other_intro)+'</p>'+table(other_headers, other_rows)+
               table(data['other_failure_headers'], data['other_failures'])+'</details>')
    combined_headers = ['Model', 'Backend', 'Cache K/V', 'FA', 'MTP',
                        'Allocated / usable per user', 'Total allocated / usable',
                        'GPU GiB', 'GPU spare MiB', 'Result']
    combined = [[*r[:6], f'{r[6]} / {r[7]}', r[8], r[11].split(' / ')[0], r[9], r[10]] for r in rows]
    memory = capacity_failure_memory(run, data)
    for failure in failures:
        model, backend, users, allocated, cache, mtp, outcome = failure
        usable = allocated - (4 if mtp == 'on' else 0)
        gpu, spare = memory.get(tuple(failure[:6]), ('—', '—'))
        outcome = outcome.replace('Fail: oom', 'Fail: out of GPU memory').replace('nonresident_vram', 'GPU memory spill')
        outcome = outcome.replace('VRAM eviction', 'GPU memory spill').replace('calculated memory floor', 'exceeds GPU memory')
        combined.append([model, backend, users, cache, 'on', mtp, f'{allocated:,} / {usable:,}',
                         f'{allocated*users:,} / {usable*users:,}', gpu, spare, outcome])
    combined.sort(key=lambda r: (r[0], r[1], -int(r[6].split(' / ')[0].replace(',', '')),
                                ['f16/f16', 'q8_0/q8_0', 'q4_0/q4_0'].index(r[3])))
    settings = ('<p><strong>Shared settings:</strong></p><ul>'
                '<li>Context target per user: E4B 131,072; Gemma 12B and Qwen 262,144</li>'
                '<li>Flash Attention (FA): on</li><li>Unified KV cache: on</li><li>Thinking: off</li>'
                '<li>MTP: up to three draft tokens; four tokens reserved per user when enabled</li></ul>'
                '<p>Pass: every user fills the context and generates 128 tokens, with at least 512 MiB GPU spare, '
                'no dropped input, prompt-cache reuse or context shifting.</p>'
                '<p>Allocated / usable means reserved space / input and output space. '
                'GPU memory is measured unless marked calculated; calculated values are minimum requirements. — means unavailable.</p>')
    grouped = ''.join(f'<h4 id="capacity-{n}">{n} {"user" if n == 1 else "users"}</h4>'+
                      table(combined_headers, [r[:2]+r[3:] for r in combined if r[2] == n]) for n in [1, 2, 4])
    section = '<h3>1.4 Full-context measurements</h3><p id="context-limits">'+esc(intro)+'</p>'+settings+grouped+details
    if html_only:
        return section
    markdown_intro = ('These settings passed a full text fill in every slot with 512 MiB GPU spare. '
                      'Requested maxima were 131072 tokens per user for E4B and 262144 for Gemma 12B and Qwen. '
                      'Allocated is the KV pool; usable includes input and output. MTP drafts at most three tokens '
                      'and reserves four tokens per user. Each slot generated 128 tokens with no input dropped, '
                      'prompt-cache reuse or context shifting. FA and unified KV are on. '
                      'Reduced contexts are passing settings, not exact maxima. Images and speed were not retested.')
    benchmark = repo/'BENCHMARK.md'
    text = benchmark.read_text().split('## 6. Cache capacity')[0].rstrip()
    text += '\n\n## 6. Cache capacity\n\n'+summary+'\n\n'+markdown_intro+'\n\n'+markdown(headers, rows)+'\n\n'+notes
    text += '\n\n### 6.1 Cache failures and calculated exclusions\n\n'+markdown(failure_headers, failures)
    text += ('\n\n### 6.2 Other measured contexts\n\n'+other_intro+
             ' GPU, process RAM and GTT are separate readings; GTT can overlap RAM.\n\n'+
             markdown(other_headers, other_rows))
    text += '\n\n'+markdown(data['other_failure_headers'], data['other_failures'])+'\n'
    benchmark.write_text(text)
    page = repo/'reports/mtp.html'
    text = page.read_text()
    start, end = text.index('<h3>1.4 Full-context measurements'), text.index('<h3>1.5 Image answers')
    text = text[:start]+section+text[end:]
    text = re.sub(r'(<p>Matched results: \d+/\d+\. )[^<]*</p>',
                  r'\g<1>Capacity: '+esc(summary)+'</p>', text, count=1)
    page.write_text(format_mtp_html(text))
    print(summary)


def render_capacity(run, repo):
    """Update capacity sections only; preserve all speed tables and image answers."""
    if (run/'capacity-summary.json').exists():
        return render_capacity_summary(run, repo)
    import html
    load = lambda p: json.loads(p.read_text())
    esc = lambda v: html.escape(str(v))
    manifest = load(run/'manifest.json')
    entries = {}
    reused = load(run/'reuse.json')
    for saved in reused:
        path = Path(saved['summary'])
        entries[saved['id']] = (load(path), path.parent)
    for selected in manifest['capacity_profiles']:
        key = f"{selected['model']}-{selected['backend']}-n{selected['slots']}-mtp{int(selected['mtp'])}-{selected['cache']}"
        path = run/'node2/capacity'/key
        saved = path/'summary.json'
        if not saved.exists():
            previous = sorted(path.glob('summary-before-*.json'), key=lambda p: p.stat().st_mtime)
            if previous:
                saved = previous[-1]
        entries[key] = (load(saved) if saved.exists() else
                       dict(selected, status='pending', largest_passing_per_user=None,
                            practical_per_user=None, guard_tokens_per_user=4 if selected['mtp'] else 0), path)
    labels = {'gemma12-Q4_0': 'Gemma 12B Q4_0', 'gemma-e4b-Q4_0': 'Gemma E4B Q4_0',
              'qwen35-MTP-Q4_K_M': 'Qwen 9B MTP Q4_K_M'}
    headers = ['Model', 'Backend', 'Users', 'Cache K/V', 'FA', 'MTP',
               'Largest resident allocated / usable per user', 'Largest resident allocated / usable total',
               'Practical allocated / usable per user',
               'Practical allocated / usable total', 'GPU spare MiB',
               'Largest-pass peak GPU / RAM / GTT GiB', 'Practical peak GPU / RAM / GTT GiB',
               'Full KV MiB', 'Full prompt / generation t/s/user',
               'Upper bound per user', 'Status']
    rows, paged = [], []
    counts = {'passed': 0, 'partial': 0, 'failed': 0, 'unsupported': 0, 'excluded': 0, 'pending': 0}
    number = lambda v: f'{v:,}' if v is not None else '—'
    for key, (item, path) in sorted(entries.items()):
        limit, practical = item['largest_passing_per_user'], item['practical_per_user']
        points = list(item.get('points', []))
        for previous in path.glob('summary-before-*.json'):
            points.extend(load(previous).get('points', []))
        guard, slots = item.get('guard_tokens_per_user', 4 if item['mtp'] else 0), item['slots']
        pair = lambda c, n=1: f'{number(c*n)} / {number((c-guard)*n)}' if c else '—'
        peaks = dict(vram_mib=0, rss_mib=0, gtt_mib=0)
        largest_peaks = dict(peaks)
        practical_free = None
        full_dir = item.get('full_result_dir', 'full-guard-'+str(guard))
        # Older completed MTP-off measurements used the directory name full.
        if practical and not (path/f'ctx-{practical}'/full_dir).exists():
            full_dir = 'full'
        memory_context = practical or limit
        full_kv, full_speed = '—', '—'
        if limit:
            full_path = path/f'ctx-{limit}'/full_dir
            if not full_path.exists():
                full_path = path/f'ctx-{limit}'/'full'
            trial = load(full_path/'trial.json')
            for f in [full_path/'trial.json', path/f'ctx-{limit}'/'image-memory/trial.json']:
                if f.exists():
                    for field in largest_peaks:
                        largest_peaks[field] = max(largest_peaks[field], load(f).get('peak_memory_mib', {}).get(field, 0))
            kv_values = [float(v) for line in trial.get('allocations', [])
                         for v in re.findall(r'(?:ROCm|Vulkan)\d+ KV buffer size\s*=\s*([\d.]+) MiB', line)]
            full_kv = f'{sum(kv_values):.0f}'
            timings = [r.get('response', {}).get('timings', {}) for r in load(full_path/'result.json')['wave']['requests']]
            rates = []
            for metric in ['prompt_per_second', 'predicted_per_second']:
                values = [t[metric] for t in timings if metric in t]
                rates.append(f'{sum(values)/len(values):.2f}' if values else '—')
            full_speed = ' / '.join(rates)
        if memory_context:
            for kind in [full_dir, 'image-memory']:
                f = path/f'ctx-{memory_context}'/kind/'trial.json'
                if f.exists():
                    trial = load(f)
                    for field in peaks:
                        peaks[field] = max(peaks[field], trial.get('peak_memory_mib', {}).get(field, 0))
            if practical and peaks['vram_mib']:
                practical_free = item.get('gpu_budget_mib', 16368)-peaks['vram_mib']
        else:
            for f in path.glob('ctx-*/full*/trial.json'):
                for field, value in load(f).get('peak_memory_mib', {}).items():
                    if field in peaks:
                        peaks[field] = max(peaks[field], value)
        status = item.get('status', 'passed' if practical else 'failed')
        if limit and not item.get('boundary_resolved') and item.get('search_mode') != 'safe_limit':
            status = 'partial'
        if not limit and status == 'failed':
            outcomes = [p.get('full', p.get('probe')) for p in item.get('points', [])]
            if outcomes and outcomes[-1] == 'unsupported':
                status = 'unsupported'
            elif outcomes and all(v == 'estimated_exclusion' for v in outcomes):
                status = 'excluded'
        counts[status if status in counts else 'failed'] += 1
        maximum = next(m['max_context'] for m in manifest['models'] if m['id'] == item['model'])
        reason = ('model limit' if limit == maximum else 'safe limit; maximum not searched' if practical and item.get('search_mode') == 'safe_limit'
                  else '4K boundary' if item.get('boundary_resolved')
                  else 'lower bound' if limit else status)
        failure_names = {'device_lost': 'device lost', 'nonresident_vram': 'VRAM eviction',
                         'unsupported': 'unsupported', 'fa_assertion': 'Flash Attention assertion',
                         'runtime': 'runtime error', 'request_error': 'request failed',
                         'context_limit': 'incomplete token counts', 'oom': 'out of GPU memory',
                         'host_backed': 'residency unconfirmed', 'estimated_exclusion': 'estimated over budget'}
        if item.get('last_failure') and status != 'passed':
            outcomes = [p.get('full', p.get('probe')) for p in item.get('points', [])]
            failure = next((v for v in reversed(outcomes) if v in failure_names), None)
            reason += ': '+failure_names.get(failure, 'request failed')
        upper = item.get('upper_bound_per_user')
        known_failures = [point['context'] for point in points
                          if point.get('context', 0) > (limit or 0) and
                          point.get('full', point.get('probe')) in ('oom', 'nonresident_vram', 'device_lost')]
        if known_failures:
            upper = min([upper] + known_failures) if upper else min(known_failures)
        upper_text = number(upper)
        failure_types = {'nonresident_vram': 'VRAM not resident', 'host_backed': 'residency unconfirmed',
                         'oom': 'OOM', 'estimated_exclusion': 'estimate', 'device_lost': 'device error',
                         'context_limit': 'token-count failure'}
        for point in points:
            outcome = point.get('full', point.get('probe'))
            if point.get('context') == upper and outcome in failure_types:
                upper_text = number(upper)+' ('+failure_types[outcome]+')'
        rows.append([labels[item['model']], item['backend'].upper(), slots,
                     item['cache']+'/'+item['cache'], 'on', 'on' if item['mtp'] else 'off',
                     pair(limit), pair(limit, slots), pair(practical), pair(practical, slots),
                     f'{practical_free:.0f}' if practical_free is not None else '—',
                     ' / '.join(f'{largest_peaks[k]/1024:.2f}' for k in largest_peaks) if limit else
                         (' / '.join(f'{peaks[k]/1024:.2f}' for k in peaks) if peaks['vram_mib'] else '—'),
                     ('same' if practical == limit else ' / '.join(f'{peaks[k]/1024:.2f}' for k in peaks)) if practical else '—',
                     full_kv, full_speed,
                     upper_text, reason])
        for correction in path.glob('ctx-*/full*/residency-correction.json'):
            raw = load(correction.parent/'result.json')
            if raw.get('status') == 'passed':
                paged.append(f"{labels[item['model']]} / {item['backend'].upper()}, {slots} users, "
                    f"{item['cache']}/{item['cache']}, MTP {'on' if item['mtp'] else 'off'}: "
                    f"{raw['context_per_user']:,} tokens per user completed, with GPU memory paging.")
    def table(head, values):
        head, values = gpu_only_columns(head, values)
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in head)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table></div>'
    markdown = '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])
    summary = ', '.join(f'{value} {name}' for name, value in counts.items())+f' across {len(rows)} selected profiles.'
    failure_headers = ['Model', 'Backend', 'Users', 'Cache K/V', 'FA', 'MTP',
                       'Allocated tokens', 'KV MiB', 'Peak GPU / RAM / GTT GiB', 'Result']
    failures = []
    old_runs = {Path(saved['summary']).parents[2] for saved in reused}
    failure_files = {f for old_run in old_runs
                     for f in (old_run/'capacity/qwen35-MTP-Q4_K_M-vulkan-n1-mtp0-f16').glob('ctx-*/full/result.json')}
    for f in sorted(failure_files):
        failed = load(f)
        if failed.get('status') != 'device_lost':
            continue
        trial = failed['trial']
        peak = trial.get('peak_memory_mib', {})
        kv = sum(float(v) for line in trial.get('allocations', [])
                 for v in re.findall(r'(?:ROCm|Vulkan)\d+ KV buffer size\s*=\s*([\d.]+) MiB', line))
        failures.append(['Qwen 9B MTP Q4_K_M', 'Vulkan', 1, 'f16/f16', 'on', 'off',
                         number(failed['context_per_user']), f'{kv:.0f}',
                         ' / '.join(f'{peak.get(k, 0)/1024:.2f}' for k in ['vram_mib','rss_mib','gtt_mib']),
                         'ErrorDeviceLost; not an OOM boundary'])
    failure_md = '\n'.join(['| '+' | '.join(failure_headers)+' |', '| '+' | '.join(['---']*len(failure_headers))+' |']+
        ['| '+' | '.join(map(str, row))+' |' for row in failures]) if failures else ''
    explanation = ('Limits require GPU-resident VRAM buffers throughout the full fill; VRAM paging is excluded. '
                   'Context values are exact tokens. Allocated is the KV pool; usable is the input/output budget. '
                   'MTP reserves four tokens per user and drafts at most three tokens. Off: input C − 128; '
                   'on: input C − 132; output 128 per user. FA and unified KV are on. '
                   'Practical settings leave at least 512 MiB GPU memory spare during full-fill and image-memory checks. '
                   'Memory columns distinguish largest-pass and practical peaks; without a pass, the first memory column shows the failed full-fill peak. '
                   'Full KV and rates refer to the largest passing text fill; rates are per-user means. '
                   'Capacity checks execution and memory, not long-context image quality. '
                   'Upper bounds can be tested failures or conservative memory exclusions; lower bounds are not proven maxima.')
    if manifest.get('capacity_refine') is False:
        explanation += ' Remaining profiles validate safe limits without refining the exact maximum.'
    benchmark = repo/'BENCHMARK.md'
    text = benchmark.read_text()
    text = text.split('## 6. Cache capacity')[0].rstrip()+'\n\n## 6. Cache capacity\n\n'+summary+'\n\n'+explanation+'\n\n'+markdown+'\n'
    if paged:
        text += '\n'+' '.join(paged)+'\n'
    if failures:
        text += '\n### 6.1 Vulkan device failures\n\n'+failure_md+'\n'
    benchmark.write_text(text)
    page = repo/'reports/mtp.html'
    text = page.read_text()
    start, end = text.index('<h3>1.4 Full-context measurements'), text.index('<h3>1.5 Image answers')
    text = text[:start]+'<h3>1.4 Full-context measurements</h3><p id="context-limits">'+esc(summary)+'</p><p>'+esc(explanation)+'</p>'+table(headers, rows)+text[end:]
    if paged:
        text = text.replace('<h3>1.5 Image answers', '<p>'+esc(' '.join(paged))+'</p><h3>1.5 Image answers', 1)
    if failures:
        text = text.replace('<h3>1.5 Image answers', '<details><summary>Vulkan device failures</summary>'+table(failure_headers, failures)+'</details><h3>1.5 Image answers', 1)
    text = re.sub(r'(<p>Matched results: \d+/\d+\. )[^<]*</p>',
                  r'\g<1>Capacity: '+esc(summary)+'</p>', text, count=1)
    page.write_text(format_mtp_html(text))
    (run/'capacity-tables.md').write_text(summary+'\n\n'+markdown+'\n')
    print(summary)

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path,nargs='?')
parser.add_argument('--capacity-run',type=Path,help='Update capacity sections only, preserving measured speed and image results')
parser.add_argument('--mtp-run',type=Path,help='Render matched MTP and capacity results in reports/mtp.html')
parser.add_argument('--patched-run',type=Path,action='append',default=[])
parser.add_argument('--additional-run',type=Path,action='append',default=[])
parser.add_argument('--reviews',type=Path,help='Luna reference-based judgements; original checklist reviews are not used')
args=parser.parse_args()
if args.mtp_run:
    render_mtp(args.mtp_run, Path(__file__).resolve().parent.parent)
    if args.capacity_run:
        render_capacity(args.capacity_run, Path(__file__).resolve().parent.parent)
    raise SystemExit(0)
if args.capacity_run:
    render_capacity(args.capacity_run, Path(__file__).resolve().parent.parent)
    raise SystemExit(0)
if args.run is None:
    parser.error('provide a run directory or --mtp-run')
extra_runs = args.patched_run + args.additional_run
root=Path(__file__).resolve().parent
manifest=json.loads((args.run/'manifest.json').read_text())
manifests={run:json.loads((run/'manifest.json').read_text()) for run in [args.run]+extra_runs}
status=json.loads((args.run/'status.json').read_text())
review_path=args.reviews or args.run/'reviews-luna.json'
reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
reference=json.loads((root/'reference.json').read_text())
rubric=json.loads((root/'images.json').read_text())
cases=[(args.run,model,backend,cache,f"{model['id']}-{backend}-{cache}")
       for model in manifest['models'] for backend in manifest['backends'] for cache in manifest['caches']]
patched_status=[]
for run in [args.run]+extra_runs:
    if run != args.run:
        patched_status.append(json.loads((run/'status.json').read_text()))
    if run != args.run and (run/'reviews-luna.json').exists():
        reviews.update(json.loads((run/'reviews-luna.json').read_text()))
    if run in args.additional_run:
        extra_manifest=manifests[run]
        cases.extend((run,m,b,c,f"{m['id']}-{b}-{c}") for m in extra_manifest['models']
                     for b in extra_manifest['backends'] for c in extra_manifest['caches'])
    for path in sorted(run.glob('*/result.json')):
        result=json.loads(path.read_text())
        if run not in extra_runs or result.get('fa', 'on') != 'on':
            continue
        cases.append((run,result['model'],result['backend'],result['cache'],result['id']))
rows=[]
diagnostics={source:json.loads((source/'diagnostics.json').read_text())
             for source in [args.run]+extra_runs if (source/'diagnostics.json').exists()}
for source,model,backend,cache,key in cases:
    path=source/key/'result.json'
    if not path.exists() and source in args.additional_run:
        path=source/key/'partial.json'
    d=json.loads(path.read_text()) if path.exists() else {}
    context=int(d.get('context', model.get('context', manifests[source]['context'])))
    row={'id':key+('-fa72' if source in args.patched_run else '')+(f"-ctx{context}" if source in args.additional_run else ''),'model':model['id'],'family':model['family'],'quant':model['quant'],
         'patch':'Gemma + Qwen72' if source in extra_runs else 'Gemma only',
         'backend':backend,'cache':cache,'fa':d.get('fa','on'),'fallback_of':d.get('fallback_of'),'context':context,'status':d.get('status','pending'),
         'images':{},'errors':d.get('errors',[]),'ubatch':model['ubatch'],
         'image_max':model['image_max']}
    row['errors'].extend(diagnostics.get(source,{}).get(key,[]))
    if d.get('server_exit_before_cleanup') == -11:
        row['errors'].append('Server segfaulted during image inference')
    server_log=source/key/'server.log'
    log=server_log.read_text(errors='replace') if server_log.exists() else ''
    occupied = re.findall(r'\|\s+- (?:ROCm0|Vulkan0).*?\|\s+(\d+) =\s+(\d+) \+', log)
    snapshot = int(occupied[-1][0])-int(occupied[-1][1]) if occupied else None
    row['memory_mib'] = d.get('peak_memory_mib', {}).get('vram_mib', snapshot)
    row['memory_method'] = 'sampled peak' if d.get('peak_memory_mib') else 'end snapshot' if snapshot is not None else 'screening estimate'
    row['host_rss_mib'] = d.get('peak_memory_mib', {}).get('rss_mib')
    row['gtt_mib'] = d.get('peak_memory_mib', {}).get('gtt_mib')
    row['estimated_vram_mib'] = d.get('estimated_vram_mib', model.get('estimated_vram_mib',{}).get(backend,{}).get(cache))
    row['full_context'] = d.get('full_context')
    if row['status']=='pending' and row['estimated_vram_mib']:
        limit=manifests[source].get('vram_limit_mib',16384)
        if row['estimated_vram_mib'] > limit:
            row['status']='excluded'
            row['errors']=['Pre-run memory estimate exceeds the configured VRAM budget']
    memory = row['memory_mib'] or row['estimated_vram_mib']
    row['memory_class'] = ('≤8 GiB' if memory <= 8192 else '≤16 GiB' if memory <= 16384 else '>16 GiB') if memory else 'unknown'
    if 'out of memory' in log.lower() or 'CUBLAS_STATUS_ALLOC_FAILED' in log:
        row['memory_class']='not fitting'
    if 'GGML_ASSERT(max_blocks_per_sm > 0)' in log:
        row['errors'].append('HIP projector Flash Attention occupancy assertion during image encoding')
    if 'ErrorDeviceLost' in log:
        row['errors'].append('Vulkan device lost during native-context prefill; no capacity pass')
    if 'CUBLAS_STATUS_ALLOC_FAILED' in log:
        row['errors'].append('HIP GPU allocation failed while encoding the image (CUBLAS_STATUS_ALLOC_FAILED)')
    for item in d.get('benchmark',{}).get('results',[]):
        field='pp512' if item.get('n_prompt')==512 else 'tg128'
        row[field]=item.get('avg_ts')
        row[field+'_sd']=item.get('stddev_ts')
    for item in d.get('images',[]):
        if item['repetition'] != 1:
            continue
        review=reviews.get(row['id']+'/'+item['image'],{})
        sample={k:item.get(k) for k in ['text','seconds','coverage_score','criteria','finish_reason','usage','error']}
        # Re-score saved text when the checklist matcher is corrected; never rerun inference.
        spec=next(i for i in rubric if i['id']==item['image'])
        sample['criteria']=[{'criterion':label,'weight':weight,'matched':bool(re.search(pattern,sample.get('text') or '',re.I))}
                            for label,weight,pattern in spec['criteria']]
        sample['coverage_score']=sum(c['weight'] for c in sample['criteria'] if c['matched'])
        if sample.get('error'):
            sample['coverage_score']=None
            sample['criteria']=[]
        sample['generation_ts']=item.get('timings',{}).get('predicted_per_second')
        sample['end_to_end_ts']=item.get('end_to_end_tokens_per_second')
        sample['review']=review
        if review and sample.get('text') and not sample.get('error'):
            if review.get('answer_sha256') != hashlib.sha256(sample['text'].encode()).hexdigest():
                raise ValueError(f'Luna judgement does not match saved answer: {key}/{item["image"]}')
            dimensions=reference['rubric']['dimensions']
            if any(not isinstance(review.get(k), (int,float)) or not 0 <= review[k] <= maximum
                   for k,maximum in dimensions.items()):
                raise ValueError(f'Invalid Luna judgement: {key}/{item["image"]}')
            sample['quality']=sum(review[k] for k in dimensions)
        row['images'][item['image']]=sample
    rows.append(row)
# Later patched runs replace superseded cases; never import the parked tables.
latest={}
for row in rows:
    latest[(row['model'], row['backend'], row['cache'], row['context'], row['fa'])]=row
rows=list(latest.values())
# Share exact answer/review matches; keep every run's measurements on its sample.
answers=[]
answer_ids={}
for row in rows:
    for image,sample in row['images'].items():
        if not sample.get('text'):
            continue
        answer={field:sample[field] for field in ('text','review') if field in sample}
        key=(image,json.dumps(answer,sort_keys=True,ensure_ascii=False))
        if key not in answer_ids:
            answer_ids[key]=len(answers)
            answers.append(answer)
        sample['answer_id']=answer_ids[key]
        for field in answer:
            del sample[field]
data={'status':status,'rows':rows,
      'rubric':rubric,'reference':reference,'patched_status':patched_status,'answers':answers}
encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
template=(root/'report.template.html').read_text()
reports=root.parent/'reports'
reports.mkdir(exist_ok=True)
(reports/'report.html').write_text(template.replace('__DATA__',encoded))
print(f"Report: {len(rows)} configurations; {sum(r['status']!='pending' for r in rows)} recorded; "
      f"{sum('answer_id' in s for r in rows for s in r['images'].values())} image answers; "
      f"{sum(bool(s.get('error')) for r in rows for s in r['images'].values())} images without answers; "
      f"{sum('quality' in s for r in rows for s in r['images'].values())} reviewed")
