#!/usr/bin/env python3
"""Render condensed evidence and one answer per image; raw logs remain external."""
import argparse
import hashlib
import json
import re
from pathlib import Path

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
        body = ''.join('<tr>'+''.join('<td>'+esc(fmt(v) if v is None or isinstance(v,float) else v)+'</td>' for v in row)+'</tr>' for row in values)
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in head)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    doc += '<title>AMD AI Ops — MTP and users</title><link rel="stylesheet" href="style.css"><main class="mtp-report">'
    doc += '<nav class="report-nav" aria-label="Reports"><a href="index.html">Recommendations</a><a href="report.html">Model tests</a><a href="mtp.html" aria-current="page">MTP and users</a></nav><h1>MTP, users and cache capacity</h1>'
    doc += f'<p>RX 6900 XT 16 GiB · llama.cpp <code>{esc(state["source_revision"][:9])}</code>.</p>'
    doc += f'<p>Matched results: {len(rows)}/90. Capacity checks: {len(caps)}/6 recorded, {cap_passed} validated, {len(caps)-cap_passed} unresolved.</p>'
    doc += '<h2>1. Detailed results</h2><h3>1.1 Matched tests at 16K per user</h3><details><summary>How to read the measurements</summary><p>The main matrix has 18 single-user MTP-off profiles and 72 two/four-user MTP-off/on profiles across f16/f16, Q8/Q8 and Q4/Q4 caches. Six additional single-user MTP-on profiles appear in section 1.2. Three batches per row; each request reads 512 and generates 128 tokens. Total t/s includes prompt and request time. Gen t/s/user uses server decode timing. Speed ratio is total t/s with MTP on divided by MTP off for the same profile; above 1 means faster, below 1 means slower. Each profile has four image answers once each, scored against Luna-6. See section 1.4 for full-context measurements and <a href="index.html#context-limits">Recommendations</a> for context limits.</p>'
    doc += '<p>Qwen uses the dedicated MTP-GGUF target in both modes. Gemma uses matching Q4 draft files only when MTP is on. Each image wave and timing batch starts a fresh server. Timing uses one fixed inventory/checklist task, not a broad workload sample.</p>'
    doc += '</details>'+table(headers, rows)
    if supplementary:
        doc += '<h3>1.2 Single-user MTP-on results</h3><details><summary>Show six f16/f16 profiles</summary>'+table(headers, supplementary)+'</details>'
    doc += '<h3>1.3 Ordinary llama-bench</h3><p>pp512/tg128, three repetitions, 16K allocation, microbatch 512. These measure one user with MTP off.</p>'
    doc += table(['Model / backend', 'FA', 'Cache', 'pp512 ± SD', 'tg128 ± SD'], baselines)
    doc += '<h3>1.4 Full-context measurements</h3><p>Six one-user, MTP-off, f16/f16 checks. Context values are exact tokens. Memory refers to the largest passing context, or the failed native attempt when no pass exists. Practical GPU free is the smallest sampled headroom during full-fill and image checks. Capacity does not establish long-context image quality. RAM is process RSS; GPU-accessible host memory is recorded separately in raw evidence.</p>'
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
    (repo/'reports/mtp.html').write_text(doc)
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
The four-token MTP guard would leave 16,380 usable tokens per slot, but that
full MTP capacity is not validated here. GPU and process RAM are sampled peaks.
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


parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path,nargs='?')
parser.add_argument('--mtp-run',type=Path,help='Render matched MTP and capacity results in reports/mtp.html')
parser.add_argument('--patched-run',type=Path,action='append',default=[])
parser.add_argument('--additional-run',type=Path,action='append',default=[])
parser.add_argument('--reviews',type=Path,help='Luna reference-based judgements; original checklist reviews are not used')
args=parser.parse_args()
if args.mtp_run:
    render_mtp(args.mtp_run, Path(__file__).resolve().parent.parent)
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
