"""Matched MTP/slot measurements and filled-context capacity searches."""
import concurrent.futures as cf
import statistics
from __main__ import (ROOT, M, IMAGES, BASE, save, now, request, MemoryMonitor,
                      stop, command, sp, Path, json, time, threading, urllib,
                      base64, hashlib, signal, re)

_metadata = {}
_gpu_budget = 16384


def memory_floor(model, slots, cache, context):
    """Conservative floor: target transformer weights and full-attention KV only."""
    if model['file'] not in _metadata:
        import sys
        sys.path.insert(0, str(Path(M['source'])/'gguf-py'))
        from gguf import GGUFReader
        reader = GGUFReader(model['file'])
        fields = {key: value.contents() for key, value in reader.fields.items()
                  if not key.startswith('tokenizer.')}
        arch = fields['general.architecture']
        blocks = int(fields[f'{arch}.block_count'])
        nextn = int(fields.get(f'{arch}.nextn_predict_layers', 0))
        pattern = fields.get(f'{arch}.attention.sliding_window_pattern')
        interval = fields.get(f'{arch}.full_attention_interval')
        heads = fields[f'{arch}.attention.head_count_kv']
        shared = int(fields.get(f'{arch}.attention.shared_kv_layers', 0))
        full_layers = [i for i in range(blocks-nextn-shared)
                       if (not pattern or not pattern[i]) and (not interval or (i+1) % interval == 0)]
        width = fields[f'{arch}.attention.key_length'] + fields[f'{arch}.attention.value_length']
        kv_bytes = sum((heads[i] if isinstance(heads, list) else heads)*width*2 for i in full_layers)
        weights = sum(t.n_bytes for t in reader.tensors if t.name.startswith('blk.')
                      and int(t.name.split('.')[1]) < blocks-nextn)
        _metadata[model['file']] = {'weights_bytes': weights, 'kv_bytes_per_token_f16': kv_bytes,
                                   'context_max': fields[f'{arch}.context_length']}
        save(ROOT/'capacity-memory-metadata.json', _metadata)
    info = _metadata[model['file']]
    scale = {'f16': 1, 'q8_0': 34/64, 'q4_0': 18/64}[cache]
    return (info['weights_bytes'] + info['kv_bytes_per_token_f16']*context*slots*scale)/2**20


def server_args(model, backend, slots, mtp, cache, context, fa='on'):
    args = [f"{M['source']}/build-{backend}/bin/llama-server",
            '--model', model['file'], '--mmproj', model['mmproj'],
            '--alias', model['id'], '--ctx-size', str(context * slots),
            '--parallel', str(slots), '--kv-unified',
            '--kv-unified-per-slot', str(context), '--no-context-shift',
            '--cache-ram', '0', '--ctx-checkpoints', '0',
            '--flash-attn', fa, '--cache-type-k', cache, '--cache-type-v', cache,
            '--n-gpu-layers', 'all', '--fit', 'off', '--threads', '10',
            '--batch-size', '2048', '--ubatch-size', str(model['ubatch']),
            '--image-min-tokens', str(model['image_min']),
            '--image-max-tokens', str(model['image_max']),
            '--host', '127.0.0.1', '--port', str(M['port']), '--slots',
            '--jinja', '--reasoning-budget', '0', '--log-verbosity', '4',
            '--spec-type', 'draft-mtp' if mtp else 'none']
    if mtp:
        args += ['--spec-draft-n-max', '3', '--spec-draft-ngl', 'all',
                 '--spec-draft-type-k', cache, '--spec-draft-type-v', cache]
        if model.get('draft'):
            args += ['--spec-draft-model', model['draft']]
    return args


class Trial:
    def __init__(self, args, dest):
        self.dest, self.proc, self.monitor = dest, None, None
        dest.mkdir(parents=True, exist_ok=True)
        self.result = {'command': args, 'start_utc': now()}

    def __enter__(self):
        self.log = open(self.dest/'server.log', 'w')
        self.proc = sp.Popen(self.result['command'], stdout=self.log, stderr=sp.STDOUT,
                             start_new_session=True)
        self.result['pid'] = self.proc.pid
        self.monitor = MemoryMonitor(self.proc, self.dest)
        deadline = time.monotonic() + 300
        try:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'server exited {self.proc.returncode}')
                try:
                    if request('/health', timeout=2).get('status') == 'ok':
                        self.result['props'] = request('/props')
                        return self
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            raise TimeoutError('server health timeout')
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.monitor:
            self.result['peak_memory_mib'] = self.monitor.finish()
        stop(self.proc)
        self.log.close()
        log = (self.dest/'server.log').read_text(errors='replace')
        self.result.update(stop_utc=now(), error=str(exc) if exc else None,
                           returncode=self.proc.returncode)
        self.result['draft_acceptance'] = [dict(accepted=int(a), generated=int(g)) for a, g in
            re.findall(r'draft acceptance\s*=\s*[\d.]+\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated', log)]
        self.result['allocations'] = [line for line in log.splitlines() if
            ('buffer size' in line or 'KV self size' in line or
             'n_ctx_slot =' in line or 'flash_attn =' in line)]
        layers = re.findall(r'offloaded (\d+)/(\d+) layers', log)
        self.result['full_offload'] = bool(layers) and all(a == b for a, b in layers)
        self.result['unified'] = "kv_unified = 'true'" in log
        attention = re.findall(r'flash_attn\s*=\s*(\w+)', log)
        self.result['fa_on'] = bool(attention) and all(v == 'enabled' for v in attention)
        self.result['projector_gpu'] = bool(re.search(r'CLIP using (ROCm|Vulkan)', log))
        low = log.lower()
        self.result['failure_kind'] = ('oom' if any(s in low for s in
            ['out of memory', 'failed to allocate', 'out_of_device_memory', 'not enough memory']) else
            'unsupported' if any(s in low for s in ['not supported', 'unsupported', 'invalid argument',
                                                    'quantized v cache requires flash_attn']) else
            'fa_assertion' if 'GGML_ASSERT(max_blocks_per_sm > 0)' in log else
            'runtime' if exc else None)
        if exc is None:
            self.result['failure_kind'] = None
        save(self.dest/'trial.json', self.result)


def streamed(body, timeout):
    started = time.monotonic()
    req = urllib.request.Request(BASE+'/completion',
        data=json.dumps(dict(body, stream=True)).encode(),
        headers={'Content-Type': 'application/json'})
    final, text, ttft = {}, [], None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                break
            item = json.loads(line[6:])
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            if item.get('content'):
                if ttft is None:
                    ttft = time.monotonic() - started
                text.append(item['content'])
            if item.get('stop'):
                final = item
    return {'seconds': time.monotonic()-started, 'ttft_seconds': ttft,
            'text': ''.join(text), 'response': final}


def wave(bodies, dest, image=False, timeout=1800):
    barrier = threading.Barrier(len(bodies))
    done, occupancy = threading.Event(), []
    def sample():
        while not done.is_set():
            try:
                slots = request('/slots', timeout=2)
                occupancy.append({'time': time.monotonic(), 'slots': slots})
            except (OSError, ValueError):
                pass
            done.wait(.1)
    def one(item):
        index, body = item
        barrier.wait()
        started = time.monotonic()
        try:
            if image:
                response = request('/v1/chat/completions', body, timeout)
                result = {'seconds': time.monotonic()-started, 'response': response,
                          'text': response['choices'][0]['message'].get('content') or ''}
            else:
                result = streamed(body, timeout)
            result['index'] = index
        except Exception as exc:
            result = {'index': index, 'error': str(exc), 'seconds': time.monotonic()-started}
        save(dest/f'request-{index}.json', result)
        return result
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with cf.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            responses = list(pool.map(one, enumerate(bodies)))
    finally:
        done.set()
        thread.join()
    summary = {'seconds': time.monotonic()-started, 'requests': responses,
               'max_active_slots': max([sum(bool(s.get('is_processing')) for s in p['slots'])
                                        for p in occupancy] or [0])}
    save(dest/'occupancy.json', occupancy)
    save(dest/'wave.json', summary)
    return summary


def token_prompts(n, count):
    bodies = []
    for i in range(n):
        prompt = request('/apply-template', {'messages': [{'role': 'user', 'content':
            f'User {i}, inventory code CEDAR-{731+i}. The workshop stores blue screws, red bolts, '
            'green washers and brass brackets. Write a detailed numbered checklist for organizing '
            'the workshop safely. Explain at least twenty practical steps.'}],
            'chat_template_kwargs': {'enable_thinking': False}})['prompt']
        frame = request('/tokenize', {'content': prompt, 'add_special': False,
                                      'parse_special': True})['tokens']
        seed = request('/tokenize', {'content': f' Inventory {i}: blue screws, red bolts, green washers. ',
                                    'add_special': False})['tokens']
        padding = count-len(frame)
        if padding < 0:
            raise ValueError('requested input shorter than the chat template')
        tokens = frame[:10] + (seed * (padding//len(seed)+1))[:padding] + frame[10:]
        bodies.append({'prompt': tokens, 'id_slot': i, 'n_predict': 128,
                       'temperature': 0, 'seed': 42, 'ignore_eos': True,
                       'cache_prompt': False})
    return bodies


def validate_wave(value, count, slots):
    errors = []
    for item in value['requests']:
        if item.get('error'):
            errors.append(f"request {item['index']}: {item['error']}")
            continue
        response = item.get('response', {})
        timing = response.get('timings', {})
        if (response.get('tokens_evaluated') != count or response.get('tokens_predicted') != 128
                or timing.get('prompt_n') != count or timing.get('cache_n') != 0):
            errors.append(f"request {item['index']}: incorrect counts/cache: {item.get('error') or timing}")
        if response.get('id_slot') != item['index'] or f"User {item['index']}" not in response.get('prompt', ''):
            errors.append(f"request {item['index']}: slot or input marker mismatch")
    if value['max_active_slots'] != slots:
        errors.append('simultaneous server slot occupancy not confirmed')
    return errors


def profile(model, backend, slots, mtp, cache, context, dest, kind, repetition=0, fa='on'):
    dest.mkdir(parents=True, exist_ok=True)
    saved = dest/'result.json'
    if saved.exists():
        return json.loads(saved.read_text())
    result = {'model': model['id'], 'backend': backend, 'slots': slots, 'mtp': mtp,
              'cache': cache, 'context_per_user': context, 'context': context*slots,
              'fa': fa, 'kind': kind, 'errors': [], 'start_utc': now()}
    guard = int(M.get('mtp_guard_tokens', 4)) if mtp and kind == 'capacity' else 0
    result.update(guard_tokens_per_user=guard, allocated_context_per_user=context,
                  usable_context_per_user=context-guard)
    if kind == 'probe':
        result['memory_floor_mib'] = memory_floor(model, slots, cache, context)
        if result['memory_floor_mib'] > _gpu_budget:
            result.update(status='estimated_exclusion', errors=['Target weights and full-attention KV alone exceed 16 GiB'])
            save(saved, result)
            return result
    trial = Trial(server_args(model, backend, slots, mtp, cache, context, fa), dest)
    try:
        with trial:
            result['loaded'] = True
            if kind in ('speed', 'capacity'):
                count = 512 if kind == 'speed' else context-guard-128
                result['input_tokens_per_user'] = count
                measured = wave(token_prompts(slots, count), dest,
                                timeout=M.get('full_context_timeout', 3600))
                result['wave'] = measured
                result['errors'] += validate_wave(measured, count, slots)
                if not result['errors']:
                    result['aggregate_tps'] = slots*128/measured['seconds']
                    result['per_user_tps'] = [128/r['seconds'] for r in measured['requests']]
            elif kind == 'images':
                offset = repetition * slots
                bodies = []
                for image in IMAGES[offset:offset+slots]:
                    encoded = base64.b64encode((ROOT/'media'/image['file']).read_bytes()).decode()
                    bodies.append({'model': model['id'], 'messages': [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+encoded}},
                        {'type': 'text', 'text': image['prompt']}]}], 'temperature': .2, 'seed': 42,
                        'max_tokens': 512, 'cache_prompt': False,
                        'chat_template_kwargs': {'enable_thinking': False}})
                result['wave'] = wave(bodies, dest, image=True)
                for image, response in zip(IMAGES[offset:offset+slots], result['wave']['requests']):
                    response['image'] = image['id']
                    if response.get('error'):
                        result['errors'].append(response['error'])
                if result['wave']['max_active_slots'] != len(bodies):
                    result['errors'].append('simultaneous image slot occupancy not confirmed')
                save(dest/'wave.json', result['wave'])
            elif kind == 'image-memory':
                image = IMAGES[0]
                encoded = base64.b64encode((ROOT/'media'/image['file']).read_bytes()).decode()
                body = {'model': model['id'], 'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+encoded}},
                    {'type': 'text', 'text': image['prompt']}]}], 'temperature': .2, 'seed': 42,
                    'max_tokens': 128, 'cache_prompt': False,
                    'chat_template_kwargs': {'enable_thinking': False}}
                bodies = []
                for i in range(slots):
                    item = json.loads(json.dumps(body))
                    item['messages'][0]['content'][1]['text'] = f'User {i}, CEDAR-{731+i}: '+image['prompt']
                    bodies.append(item)
                result['wave'] = wave(bodies, dest, image=True)
                result['errors'] += [r['error'] for r in result['wave']['requests'] if 'error' in r]
                if result['wave']['max_active_slots'] != slots:
                    result['errors'].append('simultaneous image slot occupancy not confirmed')
            elif kind == 'probe':
                # Load and allocate graph buffers; capacity is validated separately.
                measured = wave(token_prompts(slots, 512), dest)
                result['errors'] += validate_wave(measured, 512, slots)
                result['wave'] = measured
            result['health_after'] = request('/health', timeout=5)
            if result['health_after'].get('status') != 'ok':
                result['errors'].append('server unhealthy after workload')
    except Exception as exc:
        result['errors'].append(str(exc))
    result['trial'] = trial.result
    if result.get('loaded'):
        for flag in ['full_offload', 'unified', 'projector_gpu']:
            if not trial.result.get(flag):
                result['errors'].append(f'{flag} not confirmed')
        if fa == 'on' and not trial.result.get('fa_on'):
            result['errors'].append('FA on not confirmed')
        if kind == 'capacity':
            host_compute = max([float(value) for line in trial.result.get('allocations', [])
                                for value in re.findall(r'\w+_Host\s+compute buffer size\s*=\s*([\d.]+) MiB', line)] or [0])
            gtt = trial.result.get('peak_memory_mib', {}).get('gtt_mib', 0)
            result['logged_host_compute_mib'] = host_compute
            if gtt-host_compute > 1024:
                result['errors'].append('unexplained GTT exceeds logged host compute buffers by 1 GiB: GPU residency not established')
        if mtp:
            if kind in ('speed', 'capacity', 'probe'):
                drafted = bool(result.get('wave', {}).get('requests')) and all(
                    r.get('response', {}).get('timings', {}).get('draft_n', 0) > 0
                    for r in result['wave']['requests'])
            else:
                drafted = any(v['generated'] > 0 for v in trial.result.get('draft_acceptance', []))
            if not drafted:
                result['errors'].append('MTP draft execution not confirmed for every text request or image wave')
    result['status'] = 'passed' if not result['errors'] else trial.result.get('failure_kind') or 'failed'
    if result['errors']:
        detail = (' '.join(result['errors'])+' '+(dest/'server.log').read_text(errors='replace')).lower()
        if 'errordevicelost' in detail or 'device lost' in detail:
            result['status'] = 'device_lost'
        elif any(r.get('error') for r in result.get('wave', {}).get('requests', [])):
            result['status'] = trial.result.get('failure_kind') or 'request_error'
    if kind == 'capacity' and result['status'] == 'failed':
        if any('incorrect counts/cache' in error for error in result['errors']):
            result['status'] = 'context_limit'
        elif any('GPU residency not established' in error for error in result['errors']):
            result['status'] = 'host_backed'
    result['stop_utc'] = now()
    save(saved, result)
    print(f"{dest.relative_to(ROOT)}: {result['status']}", flush=True)
    if backend == 'hip' and fa == 'on' and result['status'] == 'fa_assertion':
        diagnostic_kind = 'images' if kind == 'images' else 'probe'
        profile(model, backend, slots, mtp, cache, context,
                dest/'diagnostic-fa-off', diagnostic_kind, repetition, fa='off')
        if cache != 'f16':
            profile(model, backend, slots, mtp, 'f16', context,
                    dest/'diagnostic-fa-off-f16', diagnostic_kind, repetition, fa='off')
    return result


def run():
    global _gpu_budget
    totals = list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total'))
    _gpu_budget = min(16384, sum(int(p.read_text()) for p in totals)/2**20)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    caches = M.get('caches', ['f16', 'q8_0', 'q4_0'])
    status_file = ROOT/'status-mtp.json'
    previous = json.loads(status_file.read_text()) if status_file.exists() else {}
    state = {'start_utc': previous.get('start_utc', now()), 'pid': __import__('os').getpid(),
             'status': 'running', 'matched': [], 'capacity': [],
             'matched_total': len(M['models'])*len(M['backends'])*5*len(caches),
             'capacity_total': len(M['models'])*len(M['backends']) if M.get('capacity', True) else 0}
    active = previous.get('service_was_active',
        sp.run(['systemctl', 'is-active', '--quiet', M['service']]).returncode == 0)
    state['service_was_active'] = active
    state['source_revision'] = sp.check_output(['git', '-C', M['source'], 'rev-parse', 'HEAD'], text=True).strip()
    state['service_sha256'] = hashlib.sha256(Path('/etc/systemd/system/'+M['service']).read_bytes()).hexdigest()
    save(status_file, state)
    try:
        sp.run(['sudo', 'systemctl', 'stop', M['service']], check=True)
        state['phase'] = 'capability'
        state['capability'] = []
        for backend in M['backends']:
            deadline = time.monotonic() + 1800
            while not Path(f"{M['source']}/build-{backend}/bin/llama-server").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{backend} evaluation build not ready')
                time.sleep(5)
            for model in M['models']:
                for slots in [2, 4]:
                    key = f"{model['id']}-{backend}-n{slots}-mtp1"
                    state['current'] = key
                    save(status_file, state)
                    dest = ROOT/'matched'/key
                    speed = profile(model, backend, slots, True, 'f16', 16384,
                                    dest/'speed-1', 'speed')
                    image = None
                    if speed['status'] == 'passed':
                        image = profile(model, backend, slots, True, 'f16', 16384,
                                        dest/'images-1', 'images')
                    state['capability'].append({'id': key, 'text': speed['status'],
                                                'images': image['status'] if image else 'not run'})
                    save(status_file, state)
        state['phase'] = 'matched'
        for model in M['models']:
            for backend in M['backends']:
                dest = ROOT/'baseline'/f"{model['id']}-{backend}"
                dest.mkdir(parents=True, exist_ok=True)
                if not (dest/'result.json').exists():
                    args = [f"{M['source']}/build-{backend}/bin/llama-bench", '-m', model['file'],
                            '-p', '512', '-n', '128', '-r', '3', '-t', '10', '-b', '2048',
                            '-ub', '512', '-ngl', '99', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16',
                            '-fitc', '16384', '-fitt', '0', '-o', 'json', '-v']
                    save(dest/'result.json', command(args, dest/'bench.json'))
        for slots in [1, 2, 4]:
            for model in M['models']:
                for backend in M['backends']:
                    for mtp in ([False] if slots == 1 else [False, True]):
                        for cache in caches:
                            key = f"{model['id']}-{backend}-n{slots}-mtp{int(mtp)}"
                            if cache != 'f16':
                                key += '-'+cache
                            dest = ROOT/'matched'/key
                            state['current'] = key
                            save(status_file, state)
                            summary_path = dest/'summary.json'
                            if summary_path.exists():
                                state['matched'].append(json.loads(summary_path.read_text()))
                                continue
                            speeds, images = [], []
                            for rep in range(3):
                                result = profile(model, backend, slots, mtp, cache, 16384,
                                                 dest/f'speed-{rep+1}', 'speed', rep)
                                speeds.append(result)
                                if result['status'] != 'passed':
                                    break
                            if all(v['status'] == 'passed' for v in speeds):
                                for rep in range((4+slots-1)//slots):
                                    result = profile(model, backend, slots, mtp, cache, 16384,
                                                     dest/f'images-{rep+1}', 'images', rep)
                                    images.append(result)
                                    if result['status'] != 'passed':
                                        break
                            summary = {'id': key, 'model': model, 'backend': backend, 'slots': slots,
                                       'mtp': mtp, 'cache': cache, 'speeds': speeds, 'images': images,
                                       'status': 'passed' if len(speeds) == 3 and all(v['status']=='passed' for v in speeds+images)
                                          and sum(len(v['wave']['requests']) for v in images) == 4 else 'failed'}
                            if summary['status'] == 'passed':
                                values = [v['aggregate_tps'] for v in speeds]
                                summary['aggregate_tps_mean'] = statistics.mean(values)
                                summary['aggregate_tps_sd'] = statistics.stdev(values)
                            save(summary_path, summary)
                            state['matched'].append(summary)
                            save(status_file, state)
        if M.get('capacity', True):
            state['phase'] = 'capacity'
            save(status_file, state)
            for model in sorted(M['models'], key=lambda value: value['max_context']):
                for backend in sorted(M['backends']):
                    for slots in [1]:
                        for mtp in [False]:
                            for cache in ['f16']:
                                guard = int(M.get('mtp_guard_tokens', 4)) if mtp else 0
                                full_dir = f'full-guard-{guard}' if guard else 'full'
                                key = f"{model['id']}-{backend}-n{slots}-mtp{int(mtp)}-{cache}"
                                state['current'] = key
                                save(status_file, state)
                                dest = ROOT/'capacity'/key
                                if (dest/'summary.json').exists():
                                    state['capacity'].append(json.loads((dest/'summary.json').read_text()))
                                    continue
                                maximum = int(model['max_context'])
                                low, high, candidate, points = 0, maximum+4096, maximum, []
                                last_failure = []
                                while candidate >= 4096:
                                    probe = profile(model, backend, slots, mtp, cache, candidate,
                                                    dest/f'ctx-{candidate}'/'probe', 'probe')
                                    points.append({'context': candidate, 'probe': probe['status']})
                                    if probe.get('errors'):
                                        last_failure = probe['errors']
                                    if probe['status'] == 'passed':
                                        low = candidate
                                        if candidate == maximum or high-low <= 4096:
                                            break
                                    elif probe['status'] in ('oom', 'estimated_exclusion'):
                                        high = candidate
                                    else:
                                        break
                                    candidate = ((low+high)//8192)*4096 if low else (candidate//8192)*4096
                                validated, practical = None, None
                                while low >= 4096:
                                    full = profile(model, backend, slots, mtp, cache, low,
                                                   dest/f'ctx-{low}'/full_dir, 'capacity')
                                    points.append({'context': low, 'full': full['status']})
                                    if full.get('errors'):
                                        last_failure = full['errors']
                                    if full['status'] == 'passed':
                                        image = profile(model, backend, slots, mtp, cache, low,
                                                        dest/f'ctx-{low}'/'image-memory', 'image-memory')
                                        points.append({'context': low, 'image_memory': image['status']})
                                        if image.get('errors'):
                                            last_failure = image['errors']
                                        if image['status'] == 'passed':
                                            validated = low if validated is None else validated
                                            peak = max(full['trial']['peak_memory_mib'].get('vram_mib', 16384),
                                                       image['trial']['peak_memory_mib'].get('vram_mib', 16384))
                                            if peak <= _gpu_budget-512:
                                                practical = low
                                                break
                                            # Include the measured full-fill/image workspace missing from a short probe.
                                            reference_probe = profile(model, backend, slots, mtp, cache, low,
                                                                      dest/f'ctx-{low}'/'probe', 'probe')
                                            reference_peak = reference_probe.get('trial', {}).get('peak_memory_mib', {}).get('vram_mib', peak)
                                            margin_overhead = max(0, peak-reference_peak)
                                            points.append({'context': low, 'margin_overhead_mib': margin_overhead})
                                            margin_low, margin_high = 0, low
                                            while margin_high-margin_low > 4096:
                                                candidate = ((margin_low+margin_high)//8192)*4096
                                                probe = profile(model, backend, slots, mtp, cache, candidate,
                                                                dest/f'ctx-{candidate}'/'probe', 'probe')
                                                probe_peak = probe.get('trial', {}).get('peak_memory_mib', {}).get('vram_mib', 16384)
                                                if probe['status'] == 'passed' and probe_peak+margin_overhead <= _gpu_budget-512:
                                                    margin_low = candidate
                                                else:
                                                    margin_high = candidate
                                            low = margin_low
                                            continue
                                        if image['status'] != 'oom':
                                            break
                                    if full['status'] not in ('passed', 'oom', 'context_limit', 'host_backed'):
                                        break
                                    if validated is None:
                                        high = min(high, low)
                                    low -= 4096
                                summary = {'id': key, 'model': model['id'], 'backend': backend,
                                    'slots': slots, 'mtp': mtp, 'cache': cache, 'points': points,
                                    'gpu_budget_mib': _gpu_budget,
                                    'guard_tokens_per_user': guard, 'full_result_dir': full_dir,
                                    'largest_passing_per_user': validated, 'practical_per_user': practical,
                                    'largest_usable_per_user': validated-guard if validated else None,
                                    'largest_usable_total': (validated-guard)*slots if validated else None,
                                    'practical_usable_per_user': practical-guard if practical else None,
                                    'upper_bound_per_user': min(maximum, high),
                                    'boundary_resolved': validated == maximum or
                                        (validated is not None and high-validated <= 4096),
                                    'meets_16k_per_user': validated is not None and validated >= 16384,
                                    'last_failure': last_failure,
                                    'status': 'passed' if validated and practical else 'partial' if validated else 'failed'}
                                save(dest/'summary.json', summary)
                                state['capacity'].append(summary)
                                save(status_file, state)
        state['status'] = 'finished'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        if active:
            state['restore_returncode'] = sp.run(['sudo', 'systemctl', 'start', M['service']]).returncode
            config = Path('/etc/systemd/system/'+M['service']).read_text()
            port = re.search(r'--port\s+(\d+)', config)
            if port and state['restore_returncode'] == 0:
                deadline = time.monotonic()+300
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:'+port[1]+'/health', timeout=3) as response:
                            state['restore_health'] = json.load(response)
                        if state['restore_health'].get('status') == 'ok':
                            break
                    except (OSError, ValueError) as exc:
                        state['restore_health_error'] = str(exc)
                    time.sleep(2)
        state['service_sha256_after'] = hashlib.sha256(Path('/etc/systemd/system/'+M['service']).read_bytes()).hexdigest()
        state['stop_utc'] = now()
        save(status_file, state)
