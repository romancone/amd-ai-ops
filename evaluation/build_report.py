#!/usr/bin/env python3
"""Import recorded benchmarks and scores into the shared dataset."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from report_data import publish_section, update_data, refresh_comparison, merge_short_results, make_table, write_benchmark


def render_short(run, repo):
    import statistics as stats
    load = lambda path: json.loads(path.read_text())
    fmt = lambda value: f'{value:.2f}' if isinstance(value, (float, int)) else '—'
    state, identity = load(run/'selected-status.json'), load(run/'before.json')
    reference = load(repo/'evaluation/reference.json')
    reviews = load(run/'reviews-luna.json') if (run/'reviews-luna.json').exists() else {}
    counterparts = {item['id']: item for item in load(run/'counterparts.json')}
    records, rows, answer_views = {}, [], {}

    def metrics(item):
        requests = [r for speed in item['speeds'] for r in speed.get('wave', {}).get('requests', [])]
        timings = [r.get('response', {}).get('timings', {}) for r in requests]
        mean = lambda values: stats.mean(values) if values else None
        trials = [v['trial'] for v in item['speeds']+item['images']]
        proposed = sum(t.get('draft_n', 0) for t in timings)
        accepted = sum(t.get('draft_n_accepted', 0) for t in timings)
        images, qualities = [], []
        for wave in item['images']:
            for sample in wave.get('wave', {}).get('requests', []):
                image, text = sample.get('image'), sample.get('text', '')
                if not image or not text:
                    continue
                prompt = reference['answers'][image]['prompt']
                digest = hashlib.sha256((image+'\0'+prompt+'\0'+text).encode()).hexdigest()
                review = reviews.get(digest)
                if review:
                    if review.get('answer_sha256') != hashlib.sha256(text.encode()).hexdigest():
                        raise ValueError('Review does not match answer: '+digest)
                    if review.get('reviewer') != 'gpt-6-luna':
                        raise ValueError('Expected a Luna-6 judgement: '+digest)
                    for dimension, maximum in reference['rubric']['dimensions'].items():
                        if not isinstance(review.get(dimension), (int, float)) or not 0 <= review[dimension] <= maximum:
                            raise ValueError('Invalid rubric score: '+digest)
                    qualities.append(sum(review[k] for k in reference['rubric']['dimensions']))
                images.append(dict(image=image, prompt=prompt, text=text, digest=digest, review=review,
                                   finish_reason=sample.get('response', {}).get('choices', [{}])[0].get('finish_reason')))
        return dict(status=item['status'], speed_repetitions=len(item['speeds']), image_count=len(images),
                    ttft_seconds_mean=mean([r['ttft_seconds'] for r in requests if r.get('ttft_seconds') is not None]),
                    prefill_tps_mean=mean([t['prompt_per_second'] for t in timings if 'prompt_per_second' in t]),
                    decode_tps_mean=mean([t['predicted_per_second'] for t in timings if 'predicted_per_second' in t]),
                    end_to_end_tps_mean=item.get('aggregate_tps_mean'), end_to_end_tps_sd=item.get('aggregate_tps_sd'),
                    request_seconds_mean=mean([r['seconds'] for r in requests]),
                    draft_proposed=proposed, draft_accepted=accepted,
                    draft_acceptance_percent=100*accepted/proposed if proposed else None,
                    image_quality_percent=mean(qualities) if len(qualities)==4 else None,
                    peak_gpu_mib=max([t.get('peak_memory_mib', {}).get('vram_mib', 0) for t in trials] or [0]),
                    images=images, errors=[error for v in item['speeds']+item['images'] for error in v.get('errors', [])])

    for completed in state['completed']:
        key = completed['id']
        path = run/'matched'/key/'summary.json'
        item = load(path)
        cache = item['cache']
        if item['slots'] != 1 or not item['mtp'] or cache not in ('f16','q8_0','q4_0'):
            raise ValueError('Unexpected Short profile: '+key)
        if any(v['context_per_user'] != 16384 or not v['kv_unified'] or v['fa'] != 'on'
               or v['guard_tokens_per_user'] != 4 for v in item['speeds']+item['images']):
            raise ValueError('Short profile settings differ: '+key)
        off_key = key.replace('-mtp1-', '-mtp0-')
        off_path = run/'counterparts'/off_key/'summary.json'
        if hashlib.sha256(off_path.read_bytes()).hexdigest() != counterparts[off_key]['sha256']:
            raise ValueError('Reused counterpart changed: '+off_key)
        off_item = load(off_path)
        if off_item['mtp'] or any(off_item[k] != item[k] for k in ('slots','cache','backend')) or off_item['model']['id'] != item['model']['id']:
            raise ValueError('Reused counterpart settings differ: '+off_key)
        off, on = metrics(off_item), metrics(item)
        baseline = off['end_to_end_tps_mean']
        ratio = on['end_to_end_tps_mean']/baseline if on['end_to_end_tps_mean'] and baseline else None
        record = dict(id=key, model=item['model'], backend=item['backend'], users=1, context_per_user=16384,
                      cache=cache, kv_unified=True, fa='on', thinking=False, draft_tokens=3, guard_tokens=4,
                      usable_context_per_user=16380, off=off, on=on, counterpart=counterparts[off_key],
                      result_path=str(path.relative_to(run)), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      restoration=completed['restoration'])
        records[key] = record
        values = [item['model']['id'], item['backend'], 1, 'on', cache+'/'+cache, 'on',
                  fmt(on['prefill_tps_mean']), fmt(on['decode_tps_mean']),
                  fmt(on['end_to_end_tps_mean'])+' ± '+fmt(on['end_to_end_tps_sd']), fmt(ratio),
                  fmt(on['ttft_seconds_mean']*1000) if on['ttft_seconds_mean'] is not None else '—',
                  fmt(on['request_seconds_mean']), fmt(on['draft_acceptance_percent']),
                  fmt(on['image_quality_percent']) if on['image_quality_percent'] is not None else 'Not scored',
                  fmt(on['peak_gpu_mib']/1024), on['status']]
        rows.append({'cells': [{'value': str(v)} for v in values]})
        for image in on['images']:
            entry = answer_views.setdefault(image['digest'], dict(image=image, configurations=[]))
            entry['configurations'].append(key)
    answer_records = []
    for entry in answer_views.values():
        image = entry['image']
        review = image['review']
        quality = sum(review[k] for k in reference['rubric']['dimensions']) if review else None
        answer_records.append(dict(label=image['image']+' · '+fmt(quality)+'%', image=image['image'],
            src='../media/qengineering/'+image['image']+'.jpg', text=image['text'],
            paragraphs=[image['prompt'], 'Finish: '+str(image['finish_reason']),
                        review.get('note', '') if review else 'Awaiting Luna-6 review',
                        'Configurations: '+', '.join(entry['configurations'])]))
    unscored = sorted(key for key, entry in answer_views.items() if not entry['image']['review'])
    payload = dict(cases=records, identity=identity, status=state['status'], start_utc=state['start_utc'],
                   stop_utc=state.get('stop_utc'), evidence=str(run.resolve()),
                   node2_evidence=state.get('node2_evidence'), restoration=state['restoration'],
                   quality_status='pending' if unscored else 'complete', unscored_answers=unscored,
                   tg_metric='Mean server predicted_per_second across three repetitions; excludes prefill',
                   speed_ratio_metric='End-to-end throughput ratio, including prefill')
    merge_short_results(repo, payload, rows, answer_records)
    scored = sum(v['on']['image_quality_percent'] is not None for v in records.values())
    print(f'Short MTP-on: {len(records)} outcomes; {scored}/{len(records)} image suites scored')

def render_mtp(run, repo):
    import statistics as stats
    load = lambda path: json.loads(path.read_text())
    fmt = lambda value: f'{value:.2f}' if isinstance(value, (float, int)) else '—'
    reference = load(repo/'evaluation/reference.json')
    reviews = load(run/'reviews-luna.json') if (run/'reviews-luna.json').exists() else {}
    state = load(run/'status-mtp.json')
    headers = ['Model', 'Backend', 'Users', 'MTP', 'K/V cache', 'FA', 'Prompt t/s/user', 'Gen t/s/user',
               'Total t/s ± SD', 'Speed ratio (on/off)', 'TTFT ms (mean)', 'Request s', 'Draft accept %',
               'Image quality %', 'Peak GPU GiB', 'Peak RAM GiB', 'Status']
    rows, answers, summaries = [], {}, {}
    for path in sorted((run/'matched').glob('*/summary.json')):
        item = load(path)
        summaries[item['id']] = item
    for key, item in summaries.items():
        if any(v.get('context_per_user') != 16384 or v.get('kv_unified') is not True
               or v.get('fa') != 'on' for v in item['speeds']+item['images']):
            raise ValueError('Matched Short results require 16K per user, unified KV and FA on: '+key)
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
                entry = answers.setdefault(digest, {'configurations': []})
                entry['configurations'].append(key)
                entry['answer'] = dict(label=image+' · '+fmt(quality)+'%', image=image,
                    src='../media/qengineering/'+image+'.jpg', text=text,
                    paragraphs=[prompt, 'Finish: '+str(sample.get('response', {}).get('choices', [{}])[0].get('finish_reason', 'unknown')),
                                review.get('note', review.get('reason', 'Awaiting Luna-6 review'))])
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
                   'Peak GPU GiB', 'Peak RAM GiB',
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
        if limit:
            for kind in [full_dir, 'image-memory']:
                trial = load(path.parent/f'ctx-{limit}'/kind/'trial.json')
                if kind == full_dir:
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
            peaks['vram_mib']/1024 if limit or failed_memory else None, peaks['rss_mib']/1024 if limit or failed_memory else None,
            peaks['gtt_mib']/1024 if limit or failed_memory else None, status])
    blocks = [make_table(headers, rows, 'MTP at 16K per user'),
              make_table(headers, supplementary, 'Single-user MTP on'),
              make_table(['Model / backend', 'FA', 'Cache', 'pp512 ± SD', 'tg128 ± SD'], baselines, 'llama-bench baselines'),
              make_table(cap_headers, caps, 'Single-user capacity')]
    for entry in answers.values():
        entry['answer']['paragraphs'].append('Configurations: '+', '.join(entry['configurations']))
        blocks.append(dict(kind='answer', answers=[entry['answer']]))
    publish_section(repo, 'matched', blocks, f"{run.resolve()}:{state['source_revision']}:{state.get('start_utc','')}")
    print(f'Imported {len(rows)} matched rows, {len(caps)} capacity curves and {len(answers)} unique answers.')


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


def render_capacity_summary(run, repo):
    """Import recorded capacity outcomes without changing speed or image results."""
    data = json.loads((run/'capacity-summary.json').read_text())
    rows, failures = data['rows'], data['failures']
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
    blocks = [make_table(combined_headers, [r[:2]+r[3:] for r in combined if r[2]==n], f'{n} users: capacity')
              for n in (1,2,4)]
    blocks.extend([make_table(data['other_headers'], data['other_rows'], 'Other measured contexts'),
                   make_table(data['other_failure_headers'], data['other_failures'], 'Other capacity failures')])
    publish_section(repo, 'capacity', blocks, run.resolve())
    print(data['summary'])


def render_capacity(run, repo):
    """Update capacity sections only; preserve all speed tables and image answers."""
    if (run/'capacity-summary.json').exists():
        return render_capacity_summary(run, repo)
    load = lambda p: json.loads(p.read_text())
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
               'Full KV MiB',
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
        full_kv = '—'
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
                     full_kv,
                     upper_text, reason])
        for correction in path.glob('ctx-*/full*/residency-correction.json'):
            raw = load(correction.parent/'result.json')
            if raw.get('status') == 'passed':
                paged.append(f"{labels[item['model']]} / {item['backend'].upper()}, {slots} users, "
                    f"{item['cache']}/{item['cache']}, MTP {'on' if item['mtp'] else 'off'}: "
                    f"{raw['context_per_user']:,} tokens per user completed, with GPU memory paging.")
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
    publish_section(repo, 'capacity', [make_table(headers, rows, 'Capacity'),
                    make_table(failure_headers, failures, 'Capacity failures')], run.resolve())
    print(summary)


def render_cached(run, repo):
    """Import recorded cached continuations without generating duplicate tables."""
    data = json.loads((run/'cached-text-results.json').read_text())
    refresh_comparison(repo, data)
    groups = ('cases', 'reruns', 'context64', 'two_user_cases', 'context256', 'extension_cases')
    count = sum(len(data.get(group, [])) for group in groups)
    print(f'Imported {count} cached-context attempts, including retained failures and reruns.')


parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path,nargs='?')
parser.add_argument('--markdown',action='store_true',help='Regenerate BENCHMARK.md from the shared dataset without importing or running tests')
parser.add_argument('--capacity-run',type=Path,help='Update capacity sections only, preserving measured speed and image results')
parser.add_argument('--cached-run',type=Path,help='Add verified cached-text-results.json to the timing reports')
parser.add_argument('--mtp-run',type=Path,help='Import matched MTP and capacity measurements into the shared dataset')
parser.add_argument('--short-run',type=Path,help='Merge selected one-user 16K MTP-on comparisons using their recorded cache types')
parser.add_argument('--patched-run',type=Path,action='append',default=[])
parser.add_argument('--additional-run',type=Path,action='append',default=[])
parser.add_argument('--reviews',type=Path,help='Luna reference-based judgements; original checklist reviews are not used')
args=parser.parse_args()
if args.markdown:
    write_benchmark(Path(__file__).resolve().parent.parent)
    raise SystemExit(0)
if args.short_run:
    render_short(args.short_run, Path(__file__).resolve().parent.parent)
    raise SystemExit(0)
if args.cached_run:
    render_cached(args.cached_run, Path(__file__).resolve().parent.parent)
    raise SystemExit(0)
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
    if model['family'] == 'qwen35-mtp':
        continue
    path=source/key/'result.json'
    if not path.exists() and source in args.additional_run:
        path=source/key/'partial.json'
    d=json.loads(path.read_text()) if path.exists() else {}
    context=int(d.get('context', model.get('context', manifests[source]['context'])))
    if context != 16384 or d.get('status') == 'excluded':
        continue
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
    if row['status'] == 'excluded':
        continue
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
    bench_results = d.get('benchmark',{}).get('results',[]) if row['context'] != 262144 else []
    for item in bench_results:
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
reports=root.parent/'reports'
reports.mkdir(exist_ok=True)
update_data(root.parent, 'models', data)
print(f"Report: {len(rows)} configurations; {sum(r['status']!='pending' for r in rows)} recorded; "
      f"{sum('answer_id' in s for r in rows for s in r['images'].values())} image answers; "
      f"{sum(bool(s.get('error')) for r in rows for s in r['images'].values())} images without answers; "
      f"{sum('quality' in s for r in rows for s in r['images'].values())} reviewed")
