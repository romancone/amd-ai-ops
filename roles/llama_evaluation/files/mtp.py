"""Matched MTP/slot measurements and filled-context capacity searches."""
import concurrent.futures as cf
import statistics
from __main__ import (ROOT, M, IMAGES, BASE, save, now, request, MemoryMonitor,
                      stop, command, sp, Path, json, time, threading, urllib,
                      base64, hashlib, signal, re)

_metadata = {}
_gpu_budget = 16384
_active_proc = None


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
    kv = (['--kv-unified', '--kv-unified-per-slot', str(context)]
          if M.get('kv_unified', True) else ['--no-kv-unified'])
    args = [f"{M['source']}/build-{backend}/bin/llama-server",
            '--model', model['file'], '--mmproj', model['mmproj'],
            '--alias', model['id'], '--ctx-size', str(context * slots),
            '--parallel', str(slots), *kv, '--no-context-shift',
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
        global _active_proc
        self.log = open(self.dest/'server.log', 'w')
        self.proc = sp.Popen(self.result['command'], stdout=self.log, stderr=sp.STDOUT,
                             start_new_session=True)
        _active_proc = self.proc
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
        global _active_proc
        if self.monitor:
            self.result['peak_memory_mib'] = self.monitor.finish()
            if self.monitor.eviction:
                self.result['vram_eviction'] = self.monitor.eviction
        stop(self.proc)
        if _active_proc is self.proc:
            _active_proc = None
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
        kv = re.findall(r"kv_unified = '(true|false)'", log)
        self.result['unified'] = kv[0] == 'true' if kv else None
        attention = re.findall(r'flash_attn\s*=\s*(\w+)', log)
        self.result['fa_on'] = bool(attention) and all(v == 'enabled' for v in attention)
        self.result['projector_gpu'] = bool(re.search(r'CLIP using (ROCm|Vulkan)', log))
        low = log.lower()
        self.result['failure_kind'] = ('oom' if any(s in low for s in
            ['out of memory', 'failed to allocate', 'out_of_device_memory', 'not enough memory',
             'cublas_status_alloc_failed', 'hipblas_status_alloc_failed']) else
            'unsupported' if any(s in low for s in ['not supported', 'unsupported', 'invalid argument',
                                                    'quantized v cache requires flash_attn']) else
            'fa_assertion' if 'GGML_ASSERT(max_blocks_per_sm > 0)' in log else
            'runtime' if exc else None)
        if exc is None:
            self.result['failure_kind'] = None
        save(self.dest/'trial.json', self.result)


def completion_request(body):
    payload = {k: v for k, v in body.items() if k != 'expected_cache_tokens'}
    return urllib.request.Request(BASE+'/completion',
        data=json.dumps(dict(payload, stream=True, return_tokens=True, return_progress=True)).encode(),
        headers={'Content-Type': 'application/json'})


def streamed(body, timeout, req):
    started = time.monotonic()
    final, text, ttft, events, generated = {}, [], None, [], []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                break
            item = json.loads(line[6:])
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            progress = item.get('prompt_progress')
            if progress:
                expected_cache = body.get('expected_cache_tokens')
                if expected_cache is not None and progress['cache'] != expected_cache:
                    raise RuntimeError('cached history was not reused; refusing a full re-prefill')
                continue  # Progress events are not generated tokens.
            if item.get('tokens') or item.get('content'):
                elapsed = time.monotonic() - started
                if ttft is None:
                    ttft = elapsed
                events.append({'seconds': elapsed, 'tokens': len(item.get('tokens', []))})
                generated.extend(item.get('tokens', []))
                text.append(item.get('content', ''))
            if item.get('stop'):
                final = item
    result = {'seconds': time.monotonic()-started, 'ttft_seconds': ttft,
              'started_monotonic': started,
              'token_events': events, 'generated_tokens': generated,
              'text': ''.join(text), 'response': final}
    if len(events) > 1:
        duration = events[-1]['seconds'] - events[0]['seconds']
        tokens = sum(e['tokens'] for e in events[1:])
        if duration > 0 and tokens > 0:
            result.update(decode_seconds=duration, decode_tps=tokens/duration,
                          mean_tpot_seconds=duration/tokens)
    return result


def wave(bodies, dest, image=False, timeout=1800):
    # Serialize large histories before releasing the concurrent HTTP clients.
    prepared = [completion_request(body) for body in bodies] if not image else []
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
                result = streamed(body, timeout, prepared[index])
            result['index'] = index
        except Exception as exc:
            # Do not leave other slots running after a failed stream/cache check.
            if _active_proc and _active_proc.poll() is None:
                _active_proc.kill()
            result = {'index': index, 'error': str(exc), 'seconds': time.monotonic()-started}
        save(dest/f'request-{index}.json', result)
        return result
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.monotonic()
    expired = threading.Event()
    proc = _active_proc
    def expire():
        expired.set()
        # Socket timeouts reset on incoming data; enforce a whole-wave deadline.
        if proc and proc.poll() is None:
            proc.kill()
    watchdog = threading.Timer(timeout, expire)
    watchdog.start()
    try:
        with cf.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            responses = list(pool.map(one, enumerate(bodies)))
    finally:
        watchdog.cancel()
        done.set()
        thread.join()
    summary = {'seconds': time.monotonic()-started, 'requests': responses,
               'timed_out': expired.is_set(),
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


def agentic_prompts(n, count):
    """Render real text history and a 512-token next turn without repeating padding."""
    corpus = Path(M['text_file']).read_text()
    tokenize = lambda text: request('/tokenize', {'content': text, 'add_special': False,
                                                  'parse_special': True})['tokens']
    render = lambda messages: request('/apply-template', {'messages': messages,
                                    'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    history_tokens = request('/tokenize', {'content': corpus, 'add_special': False})['tokens']
    warm, tails = [], []
    for i in range(n):
        content = f'User {i}. Review this agent work history:\n@@HISTORY@@\nStart with one status word.'
        messages = [{'role': 'user', 'content': content}]
        prefix = render(messages)
        left, right = map(tokenize, prefix.split('@@HISTORY@@'))
        needed = count-512-len(left)-len(right)
        if needed < 0 or needed > len(history_tokens):
            raise ValueError('text_file must contain enough real text for the requested context; padding is disabled')
        tokens = left+history_tokens[:needed]+right
        next_turn = render(messages+[{'role': 'assistant', 'content': '@@ANSWER@@'},
            {'role': 'user', 'content': 'New tool output:\n@@NEW@@\nExplain the next changes and checks.'}])
        # Keep the original token prefix, including its thinking-off preamble.
        # Rendering a completed assistant turn may omit that preamble (Gemma).
        # Only append the new turn; never replace already cached history.
        _, suffix = next_turn.split('@@ANSWER@@')
        tail_left, tail_right = map(tokenize, suffix.split('@@NEW@@'))
        fresh = 511-len(tail_left)-len(tail_right)  # Plus the last, not-yet-decoded assistant token.
        if fresh < 0 or needed+fresh > len(history_tokens):
            raise ValueError('text_file is too short for the new tool output')
        warm.append({'prompt': tokens, 'id_slot': i, 'n_predict': 1, 'temperature': 0,
                     'seed': 42, 'ignore_eos': True, 'cache_prompt': False})
        tails.append(tail_left+history_tokens[needed:needed+fresh]+tail_right)
    return warm, tails


def validate_wave(value, count, slots, predictions=128, cached=0, occupancy=True,
                  output_boundary=None):
    errors = []
    if value.get('timed_out'):
        errors.append('whole-wave time limit exceeded')
    for item in value['requests']:
        if item.get('error'):
            errors.append(f"request {item['index']}: {item['error']}")
            continue
        response = item.get('response', {})
        timing = response.get('timings', {})
        cache_n, prompt_n = timing.get('cache_n', 0), timing.get('prompt_n', 0)
        cache_ok = (cache_n == count-cached and prompt_n == cached
                    if cached else cache_n == 0 and prompt_n == count)
        if (response.get('tokens_evaluated') != count or response.get('tokens_predicted') != predictions
                or not cache_ok):
            errors.append(f"request {item['index']}: incorrect counts/cache: {item.get('error') or timing}")
        # With context shifting off, this server also marks the final output
        # token at an exactly filled context as truncated. Require the complete
        # input/cache counts and streamed output before accepting that boundary.
        boundary_stop = (output_boundary is not None and count+predictions == output_boundary
                         and response.get('tokens_evaluated') == count and cache_ok
                         and response.get('tokens_predicted') == predictions
                         and response.get('tokens_cached') == output_boundary-1
                         and response.get('stop_type') == 'limit'
                         and sum(e['tokens'] for e in item.get('token_events', [])) == predictions
                         and len(item.get('generated_tokens', [])) == predictions)
        if response.get('truncated') and not boundary_stop:
            errors.append(f"request {item['index']}: truncation without a complete output boundary")
        if 'token_events' in item and sum(e['tokens'] for e in item['token_events']) != predictions:
            errors.append(f"request {item['index']}: incomplete token timing stream")
        if response.get('id_slot') != item['index'] or f"User {item['index']}" not in response.get('prompt', ''):
            errors.append(f"request {item['index']}: slot or input marker mismatch")
    if occupancy and value['max_active_slots'] != slots:
        errors.append('simultaneous server slot occupancy not confirmed')
    return errors


def profile(model, backend, slots, mtp, cache, context, dest, kind, repetition=0, fa='on', diagnostics=True):
    dest.mkdir(parents=True, exist_ok=True)
    cached_text = kind in ('filled-speed', 'cached-short')
    saved = dest/'result.json'
    if saved.exists():
        cached = json.loads(saved.read_text())
        expected_kv = M.get('kv_unified', True)
        old_command = cached.get('trial', {}).get('command', [])
        if (('--kv-unified' in old_command) != expected_kv or cached.get('kind') != kind
                or cached.get('context_per_user') != context):
            raise ValueError('Measurement settings differ from the saved result; use a new evidence directory')
        if cached_text and (
                cached.get('workload') != 'cached-text-512-v3' or
                cached.get('text_sha256') != hashlib.sha256(Path(M['text_file']).read_bytes()).hexdigest()):
            raise ValueError('Text workload changed; use a new evidence directory')
        log = dest/'server.log'
        if cached['status'] != 'passed' and log.exists() and any(
                marker in log.read_text(errors='replace').lower() for marker in
                ('cublas_status_alloc_failed', 'hipblas_status_alloc_failed')):
            cached['status'] = 'oom'
            save(dest/'classification-correction.json', {
                'source_result_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
                'status': 'oom', 'reason': 'BLAS workspace allocation failed'})
        memory = dest/'memory.jsonl'
        if kind == 'capacity' and cached['status'] == 'passed' and memory.exists():
            for line in memory.read_text().splitlines():
                sample = json.loads(line)
                missing = sample.get('vram_total_mib', 0)-sample.get('vram_resident_mib', 0)
                migrated = sample.get('gtt_resident_mib', 0)-sample.get('gtt_total_mib', 0)
                if min(missing, migrated) > 1:
                    cached['status'] = 'nonresident_vram'
                    cached['errors'] = cached.get('errors', [])+['Recorded DRM sample confirms VRAM eviction to GTT']
                    save(dest/'residency-correction.json', {
                        'source_result_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
                        'sample': sample, 'status': 'nonresident_vram'})
                    break
        return cached
    result = {'model': model['id'], 'backend': backend, 'slots': slots, 'mtp': mtp,
              'cache': cache, 'context_per_user': context, 'context': context*slots,
              'fa': fa, 'kv_unified': M.get('kv_unified', True),
              'kind': kind, 'errors': [], 'start_utc': now()}
    guard = int(M.get('mtp_guard_tokens', 4)) if mtp and kind in ('capacity', 'filled-speed', 'cached-short') else 0
    result.update(guard_tokens_per_user=guard, allocated_context_per_user=context,
                  usable_context_per_user=context-guard)
    if kind == 'probe':
        result['memory_floor_mib'] = memory_floor(model, slots, cache, context)
        if result['memory_floor_mib'] > _gpu_budget:
            result.update(status='estimated_exclusion', errors=[
                f'Target weights and full-attention KV alone exceed {_gpu_budget:.0f} MiB GPU memory'])
            save(saved, result)
            return result
    trial = Trial(server_args(model, backend, slots, mtp, cache, context, fa), dest)
    case_expired = threading.Event()
    case_limit = min(M.get('full_context_timeout', 600), 1800)
    case_deadline = time.monotonic()+case_limit
    if cached_text:
        result['case_limit_seconds'] = case_limit
    def expire_case():
        case_expired.set()
        if trial.proc and trial.proc.poll() is None:
            trial.proc.kill()
    case_timer = threading.Timer(max(0, case_deadline-time.monotonic()), expire_case)
    protocol_timer = None
    if cached_text:
        case_timer.start()
    try:
        with trial:
            result['loaded'] = True
            trial.monitor.abort_on_eviction = kind in ('capacity', 'filled-speed', 'cached-short')
            if kind in ('speed', 'capacity', 'filled-speed', 'cached-short'):
                count = 4608 if kind == 'cached-short' else 512 if kind == 'speed' else context-guard-128
                result['input_tokens_per_user'] = count
                timeout = M.get('full_context_timeout', 1800)
                if cached_text:
                    result.update(workload='cached-text-512-v3', new_input_tokens_per_user=512,
                                  cached_input_tokens_per_user=count-512, output_tokens_per_user=128,
                                  tg_measurement='all-users-decoding-overlap')
                    result['text_sha256'] = hashlib.sha256(Path(M['text_file']).read_bytes()).hexdigest()
                    # Validate the protocol first; never spend a long fill discovering a cache/API error.
                    # The short prefix must exceed Gemma's 1024-token SWA window;
                    # this build otherwise resets it even for an appended turn.
                    phases = [('protocol', 4608)]
                    if kind != 'cached-short':
                        phases.append(('measurement', count))
                    for phase, tokens in phases:
                        phase_dir = dest/phase
                        prefill_dir = phase_dir/'prefill'
                        prefill_dir.mkdir(parents=True, exist_ok=True)
                        deadline = min(case_deadline, time.monotonic()+90) if phase == 'protocol' else case_deadline
                        if phase == 'protocol':
                            protocol_timer = threading.Timer(max(0, deadline-time.monotonic()), expire_case)
                            protocol_timer.start()
                        warm, tails = agentic_prompts(slots, tokens)
                        budget = deadline-time.monotonic()
                        if budget <= 0:
                            raise TimeoutError('whole-case time limit exceeded')
                        filled = wave(warm, prefill_dir, timeout=budget)
                        result[phase] = {'prefill': filled}
                        errors = validate_wave(filled, tokens-512, slots, predictions=1, occupancy=False)
                        if errors:
                            raise RuntimeError('; '.join(errors))
                        bodies = [dict(b, prompt=b['prompt']+r['generated_tokens']+tail,
                                       n_predict=128, cache_prompt=True, expected_cache_tokens=tokens-512)
                                  for b, r, tail in zip(warm, filled['requests'], tails)]
                        save(phase_dir/'inputs.json', bodies)
                        budget = deadline-time.monotonic()
                        if budget <= 0:
                            raise TimeoutError('whole-case time limit exceeded')
                        measured = wave(bodies, phase_dir, timeout=budget)
                        result[phase]['continuation'] = measured
                        errors = validate_wave(measured, tokens, slots, cached=512, occupancy=False,
                                               output_boundary=context if not mtp else None)
                        if errors:
                            raise RuntimeError('; '.join(errors))
                        if not all(r.get('decode_tps') for r in measured['requests']):
                            raise RuntimeError('complete streamed generation timings not confirmed')
                        starts = [r['started_monotonic']+r['token_events'][0]['seconds'] for r in measured['requests']]
                        ends = [r['started_monotonic']+r['token_events'][-1]['seconds'] for r in measured['requests']]
                        overlap_start, overlap_end = max(starts), min(ends)
                        # All prefills have finished at the latest first token. Exclude
                        # first tokens and stop before any user leaves the decode batch.
                        overlap_tokens = [sum(e['tokens'] for e in r['token_events']
                                              if overlap_start < r['started_monotonic']+e['seconds'] <= overlap_end)
                                          for r in measured['requests']]
                        if overlap_end <= overlap_start or not all(overlap_tokens):
                            raise RuntimeError('overlapping token generation not confirmed for every user')
                        measured.update(decode_overlap_seconds=overlap_end-overlap_start,
                                        decode_overlap_tokens_per_user=overlap_tokens)
                        save(phase_dir/'wave.json', measured)
                        if phase == 'protocol':
                            protocol_timer.cancel()
                else:
                    measured = wave(token_prompts(slots, count), dest, timeout=timeout)
                result['wave'] = measured
                result['errors'] += validate_wave(measured, count, slots,
                                                 cached=512 if cached_text else 0,
                                                 occupancy=not cached_text,
                                                 output_boundary=context if cached_text and not mtp else None)
                if not result['errors']:
                    result['end_to_end_tps'] = slots*128/measured['seconds']
                    if kind == 'speed':  # Preserve the established 16K metric.
                        result['aggregate_tps'] = result['end_to_end_tps']
                        result['per_user_tps'] = [128/r['seconds'] for r in measured['requests']]
                    if cached_text:
                        requests = measured['requests']
                        duration = measured['decode_overlap_seconds']
                        result['decode_tps_per_user'] = [n/duration for n in measured['decode_overlap_tokens_per_user']]
                        result['decode_aggregate_tps'] = sum(result['decode_tps_per_user'])
                        result['ttft_seconds_mean'] = statistics.mean(r['ttft_seconds'] for r in requests)
                        # End-to-end throughput also excludes client file writes and monitor shutdown.
                        first_send = min(r['started_monotonic'] for r in requests)
                        last_token = max(r['started_monotonic']+r['token_events'][-1]['seconds'] for r in requests)
                        result['end_to_end_tps'] = slots*128/(last_token-first_send)
                        result['request_send_skew_seconds'] = max(r['started_monotonic'] for r in requests)-first_send
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
            if kind in ('capacity', 'filled-speed', 'cached-short'):
                result['drm_memory_after'] = trial.monitor.drm_memory()
    except Exception as exc:
        result['errors'].append(str(exc))
    finally:
        case_timer.cancel()
        if protocol_timer:
            protocol_timer.cancel()
    result['trial'] = trial.result
    if result.get('loaded'):
        for flag in ['full_offload', 'projector_gpu']:
            if not trial.result.get(flag):
                result['errors'].append(f'{flag} not confirmed')
        if trial.result.get('unified') != M.get('kv_unified', True):
            result['errors'].append('requested KV mode not confirmed')
        if fa == 'on' and not trial.result.get('fa_on'):
            result['errors'].append('FA on not confirmed')
        if kind in ('capacity', 'filled-speed', 'cached-short'):
            host_compute = max([float(value) for line in trial.result.get('allocations', [])
                                for value in re.findall(r'\w+_Host\s+compute buffer size\s*=\s*([\d.]+) MiB', line)] or [0])
            result['logged_host_compute_mib'] = host_compute
            residency = result.get('drm_memory_after', {})
            requested = residency.get('vram_total_mib')
            resident = residency.get('vram_resident_mib')
            if requested is not None and resident is not None and requested-resident > 1:
                result['errors'].append(f'{requested-resident:.2f} MiB requested VRAM is not resident: GPU residency not established')
        if mtp:
            if kind in ('speed', 'capacity', 'filled-speed', 'cached-short', 'probe'):
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
            result['status'] = 'nonresident_vram'
    if trial.result.get('vram_eviction'):
        result['errors'].append('VRAM eviction to GTT confirmed; stopped the failed fill early')
        result['status'] = 'nonresident_vram'
    if cached_text and result['status'] == 'passed':
        peak = trial.result.get('peak_memory_mib', {}).get('vram_mib')
        if peak is None or _gpu_budget-peak < 512:
            result['errors'].append('512 MiB GPU headroom not confirmed')
            result['status'] = 'headroom_failure'
    phase_waves = [w for p in ['protocol', 'measurement'] for w in result.get(p, {}).values()]
    if case_expired.is_set() or any(v.get('timed_out') for v in [result.get('wave', {}), *phase_waves]):
        result['status'] = 'timeout'
    result['stop_utc'] = now()
    save(saved, result)
    print(f"{dest.relative_to(ROOT)}: {result['status']}", flush=True)
    if diagnostics and backend == 'hip' and fa == 'on' and result['status'] == 'fa_assertion':
        diagnostic_kind = 'images' if kind == 'images' else 'probe'
        profile(model, backend, slots, mtp, cache, context,
                dest/'diagnostic-fa-off', diagnostic_kind, repetition, fa='off')
        if cache != 'f16':
            profile(model, backend, slots, mtp, 'f16', context,
                    dest/'diagnostic-fa-off-f16', diagnostic_kind, repetition, fa='off')
    return result


def capacity_curve(model, selected, dest):
    """Search one explicit profile; every reported boundary has a full fill."""
    summary_path = dest/'summary.json'
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get('kv_unified', True) != M.get('kv_unified', True):
            raise ValueError('KV mode changed; use a new evidence directory')
        return cached
    backend, slots, mtp, cache = (selected[k] for k in ('backend', 'slots', 'mtp', 'cache'))
    refine = M.get('capacity_refine', True)
    guard = int(M.get('mtp_guard_tokens', 4)) if mtp else 0
    maximum = int(model['max_context'])
    ceiling = min(maximum, selected.get('start_context', maximum))
    upper = min(maximum+4096, selected.get('failing_upper', maximum+4096))
    passing, candidate, practical = 0, ceiling, None
    points, last_failure, validated_results = [], [], {}
    runtime_failure = False
    # Narrow allocation feasibility with short probes before any expensive fill.
    probe_low, probe_high = 0, ceiling+4096
    if selected.get('reused_lower_context'):
        lower = int(selected['reused_lower_context'])
        point = dest/f'ctx-{lower}'/f'full-guard-{guard}'
        if not (point/'result.json').exists():
            raise ValueError('The reused lower bound must have a saved full-fill result')
        seed = profile(model, backend, slots, mtp, cache, lower, point, 'capacity')
        if seed['status'] != 'passed':
            raise ValueError('The reused lower bound is not a valid full fill')
        passing = probe_low = lower
        validated_results[lower] = seed
        points.append({'context': lower, 'full': 'passed', 'reused': True})
    if selected.get('start_context') and selected.get('failing_upper'):
        # Establish a lower full-fill bound before approaching a known device failure.
        point = dest/f'ctx-{candidate}'
        seed = profile(model, backend, slots, mtp, cache, candidate, point/'probe', 'probe')
        points.append({'context': candidate, 'probe': seed['status']})
        if seed['status'] == 'passed':
            seed = profile(model, backend, slots, mtp, cache, candidate,
                           point/f'full-guard-{guard}', 'capacity')
            points.append({'context': candidate, 'full': seed['status']})
        if seed['status'] == 'passed':
            passing = probe_low = candidate
            validated_results[candidate] = seed
            ceiling = min(maximum, selected['failing_upper']-4096)
            probe_high, candidate = ceiling+4096, ceiling
        elif seed['status'] not in ('oom', 'estimated_exclusion'):
            upper = min(upper, candidate)
            last_failure, runtime_failure = seed.get('errors', []), True
    while candidate >= 4096 and not runtime_failure:
        probe = profile(model, backend, slots, mtp, cache, candidate,
                        dest/f'ctx-{candidate}'/'probe', 'probe')
        points.append({'context': candidate, 'probe': probe['status']})
        reserve_ok = refine or probe.get('trial', {}).get('peak_memory_mib', {}).get('vram_mib', _gpu_budget) <= _gpu_budget-1024
        if probe['status'] == 'passed' and reserve_ok:
            probe_low = candidate
            if candidate == ceiling or probe_high-probe_low <= 4096:
                break
        elif probe['status'] == 'passed':
            probe_high = candidate
        elif probe['status'] in ('oom', 'estimated_exclusion'):
            probe_high = candidate
            upper = min(upper, candidate)
            last_failure = probe.get('errors', [])
        else:
            upper = min(upper, candidate)
            last_failure = probe.get('errors', [])
            runtime_failure = True
            break
        candidate = ((probe_low+probe_high)//8192)*4096 if probe_low else (candidate//8192)*4096
    candidate = probe_low
    while candidate >= 4096 and not runtime_failure:
        full = validated_results.get(candidate) or profile(
            model, backend, slots, mtp, cache, candidate,
            dest/f'ctx-{candidate}'/f'full-guard-{guard}', 'capacity')
        points.append({'context': candidate, 'full': full['status']})
        if full['status'] == 'passed':
            passing = candidate
            validated_results[candidate] = full
            if not refine or candidate == maximum or upper-passing <= 4096:
                break
            candidate = ((passing+upper)//8192)*4096
            continue
        last_failure = full.get('errors', [])
        upper = min(upper, candidate)
        if full['status'] not in ('oom', 'host_backed', 'nonresident_vram'):
            runtime_failure = True
            break
        candidate = ((passing+upper)//8192)*4096 if passing else (candidate//8192)*4096
    largest = passing or None
    if not refine:
        # Keep any larger completed text fill when resuming a partial search.
        for saved in dest.glob(f'ctx-*/full-guard-{guard}/result.json'):
            context = int(saved.parents[1].name.removeprefix('ctx-'))
            if context > (largest or 0) and profile(
                    model, backend, slots, mtp, cache, context, saved.parent, 'capacity')['status'] == 'passed':
                largest = context
                points.append({'context': context, 'full': 'passed', 'reused': True})
    # Validate image workspace and the 512 MiB operating margin separately.
    margin_low, margin_high = 0, passing+4096
    candidate = passing
    while candidate >= 4096:
        point = dest/f'ctx-{candidate}'
        # Reject image-allocation failures before spending time on another fill.
        image = profile(model, backend, slots, mtp, cache, candidate, point/'image-memory', 'image-memory')
        points.append({'context': candidate, 'image_memory': image['status']})
        full = None
        if image['status'] == 'passed':
            full = validated_results.get(candidate) or profile(
                model, backend, slots, mtp, cache, candidate, point/f'full-guard-{guard}', 'capacity')
        if full is not None and full['status'] == 'passed':
            peak = max(full['trial']['peak_memory_mib'].get('vram_mib', _gpu_budget),
                       image['trial']['peak_memory_mib'].get('vram_mib', _gpu_budget))
            if peak <= _gpu_budget-512:
                practical = candidate
                break
            # Estimate workspace overhead, screen a lower margin candidate, then
            # validate that candidate with a real simultaneous full fill.
            reference = profile(model, backend, slots, mtp, cache, candidate, point/'probe', 'probe')
            probe_peak = reference.get('trial', {}).get('peak_memory_mib', {}).get('vram_mib', peak)
            overhead = max(0, peak-probe_peak)
            margin_low, margin_high = 0, candidate
            while margin_high-margin_low > 4096:
                c = ((margin_low+margin_high)//8192)*4096
                probe = profile(model, backend, slots, mtp, cache, c, dest/f'ctx-{c}'/'probe', 'probe')
                points.append({'context': c, 'margin_probe': probe['status']})
                if probe['status'] not in ('passed', 'oom', 'estimated_exclusion'):
                    last_failure = probe.get('errors', [])
                    runtime_failure = True
                    break
                p = probe.get('trial', {}).get('peak_memory_mib', {}).get('vram_mib', _gpu_budget)
                if probe['status'] == 'passed' and p+overhead <= _gpu_budget-512:
                    margin_low = c
                else:
                    margin_high = c
            if runtime_failure:
                break
            candidate = margin_low
        else:
            failed = image if image['status'] != 'passed' else full
            last_failure = failed.get('errors', [])
            if failed['status'] not in ('oom', 'host_backed', 'nonresident_vram'):
                runtime_failure = True
                break
            candidate = candidate-4096 if refine else (candidate*3//16384)*4096
    resolved = not runtime_failure and (largest == maximum or
        largest is not None and upper-largest <= 4096)
    result = {'id': dest.name, 'model': model['id'], 'backend': backend, 'slots': slots,
              'mtp': mtp, 'cache': cache, 'fa': 'on', 'draft_limit': 3 if mtp else 0,
              'kv_unified': M.get('kv_unified', True),
              'guard_tokens_per_user': guard, 'gpu_budget_mib': _gpu_budget,
              'largest_passing_per_user': largest, 'practical_per_user': practical,
              'largest_usable_per_user': largest-guard if largest else None,
              'largest_usable_total': (largest-guard)*slots if largest else None,
              'practical_usable_per_user': practical-guard if practical else None,
              'practical_usable_total': (practical-guard)*slots if practical else None,
              'upper_bound_per_user': min(maximum, upper),
              'boundary_resolved': resolved,
              'search_mode': 'maximum' if refine else 'safe_limit',
              'points': points, 'last_failure': last_failure,
              'status': 'passed' if practical and (resolved or not refine) else 'partial' if largest else 'failed'}
    save(summary_path, result)
    return result


def run():
    global _gpu_budget
    if M.get('capacity_only') and not M.get('capacity_profiles'):
        raise ValueError('Capacity-only mode requires explicit capacity_profiles')
    for selected in M.get('capacity_profiles', []):
        if (selected['model'] not in [m['id'] for m in M['models']]
                or selected['backend'] not in M['backends']
                or selected['slots'] not in [1, 2, 4]
                or selected['cache'] not in ['f16', 'q8_0', 'q4_0']
                or not isinstance(selected['mtp'], bool)):
            raise ValueError(f'Invalid capacity profile: {selected}')
        model = next(m for m in M['models'] if m['id'] == selected['model'])
        if selected.get('measure') in ('filled-speed', 'cached-short') and (
                not M.get('capacity_only') or not 4096 <= selected.get('start_context', 0) <= model['max_context']):
            raise ValueError('Filled-speed needs capacity_only and an explicit context within the model limit')
        if selected.get('measure') in ('filled-speed', 'cached-short') and (
                not M.get('text_file') or not Path(M['text_file']).is_file()):
            raise ValueError('Filled-speed requires a saved text workload before the service is stopped')
    totals = list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total'))
    _gpu_budget = min(float(M.get('vram_limit_mib', 16384)),
                      sum(int(p.read_text()) for p in totals)/2**20)
    def interrupted(*_):
        # Closing the isolated server also releases pending HTTP worker threads,
        # allowing the finally block to restore the original service promptly.
        if _active_proc is not None and _active_proc.poll() is None:
            _active_proc.kill()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    caches = M.get('caches', ['f16', 'q8_0', 'q4_0'])
    matrix_models = [] if M.get('capacity_only', False) else M['models']
    profiles = []
    if M.get('capacity', True):
        profiles = M.get('capacity_profiles') or [
            {'model': model['id'], 'backend': backend, 'slots': 1,
             'mtp': False, 'cache': 'f16'}
            for model in sorted(M['models'], key=lambda v: v['max_context'])
            for backend in sorted(M['backends'])]
    status_file = ROOT/'status-mtp.json'
    previous = json.loads(status_file.read_text()) if status_file.exists() else {}
    if previous and previous.get('kv_unified', True) != M.get('kv_unified', True):
        raise ValueError('KV mode changed; use a new evidence directory')
    state = {'start_utc': previous.get('start_utc', now()), 'pid': __import__('os').getpid(),
             'status': 'running', 'matched': [], 'capacity': [], 'filled_speed': [], 'short_checks': [],
             'kv_unified': M.get('kv_unified', True),
             'matched_total': len(matrix_models)*len(M['backends'])*5*len(caches),
             'capacity_total': sum(p.get('measure', 'capacity') == 'capacity' for p in profiles),
             'filled_speed_total': sum(p.get('measure') == 'filled-speed' for p in profiles),
             'short_checks_total': sum(p.get('measure') == 'cached-short' for p in profiles)}
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
            for model in matrix_models:
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
        for model in matrix_models:
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
            for model in matrix_models:
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
        if profiles:
            state['phase'] = 'capacity'
            for selected in profiles:
                model = next(v for v in M['models'] if v['id'] == selected['model'])
                key = f"{model['id']}-{selected['backend']}-n{selected['slots']}-mtp{int(selected['mtp'])}-{selected['cache']}"
                state['current'] = key
                save(status_file, state)
                if selected.get('measure') in ('filled-speed', 'cached-short'):
                    kind = selected['measure']
                    state['phase'] = kind
                    summary = profile(model, selected['backend'], selected['slots'], selected['mtp'],
                                      selected['cache'], selected['start_context'],
                                      ROOT/kind/key/f"ctx-{selected['start_context']}",
                                      kind, diagnostics=False)
                    state['short_checks' if kind == 'cached-short' else 'filled_speed'].append(summary)
                    save(status_file, state)
                    if summary['status'] != 'passed':
                        raise RuntimeError(f'{kind} failed; stop the queue for review')
                else:
                    summary = capacity_curve(model, selected, ROOT/'capacity'/key)
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
