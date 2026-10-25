#!/usr/bin/env python3
"""Serial, resumable GPU evaluation. All raw evidence stays in the run directory."""
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess as sp
import time
import threading
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
lock = open('run.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
M = json.loads(Path('manifest.json').read_text())
IMAGES = json.loads(Path('images.json').read_text())
BASE = f"http://127.0.0.1:{M['port']}"

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def command(args, output, timeout=900):
    started = time.monotonic()
    with open(output, 'w') as out, open(str(output) + '.stderr', 'w') as err:
        proc = sp.Popen(args, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except sp.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = -124
    return {'command': args, 'returncode': rc, 'seconds': time.monotonic()-started}

def request(path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE+path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def score(text, image):
    checks = [{'criterion': label, 'weight': weight,
               'matched': bool(re.search(pattern, text, re.I))}
              for label, weight, pattern in image['criteria']]
    return {'coverage_score':sum(c['weight'] for c in checks if c['matched']),
            'criteria': checks, 'method':'automated key-fact coverage; not a hallucination score'}

def placement(log, context):
    layers = re.findall(r'offloaded (\d+)/(\d+) layers', log)
    ctx = re.findall(r'\bn_ctx\s*=\s*(\d+)', log)
    return {'layers': layers, 'context': ctx,
            'full_offload': bool(layers) and all(a == b for a,b in layers),
            'context_matches': bool(ctx) and all(int(n)==context for n in ctx),
            'flash_attention': re.findall(r'flash_attn\s*=\s*([^\n]+)', log)}

class MemoryMonitor:
    """Sample device occupancy and process resident RAM, once a second."""
    def __init__(self, proc, dest):
        self.proc, self.dest = proc, dest
        self.done = threading.Event()
        self.peak = {}
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        devices = list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total'))
        with open(self.dest/'memory.jsonl', 'w') as out:
            while not self.done.is_set():
                values = {'utc': now()}
                for path in devices:
                    for kind in ['vram', 'gtt']:
                        value = int((path.parent/f'mem_info_{kind}_used').read_text()) / 2**20
                        values[kind+'_mib'] = values.get(kind+'_mib', 0) + value
                try:
                    status = Path(f'/proc/{self.proc.pid}/status').read_text()
                    values['rss_mib'] = int(re.search(r'VmRSS:\s+(\d+)', status)[1]) / 1024
                except (OSError, TypeError):
                    pass
                for key, value in values.items():
                    if key != 'utc':
                        self.peak[key] = max(self.peak.get(key, 0), value)
                out.write(json.dumps(values)+'\n')
                out.flush()
                self.done.wait(1)

    def finish(self):
        self.done.set()
        self.thread.join()
        return self.peak

def full_context(context, dest):
    # Raw token IDs make the input length exact, independent of chat templates.
    seed = request('/tokenize', {'content': 'A landscape contains hills, trees and a river. ', 'add_special': False})['tokens']
    count = context - 128
    tokens = (seed * (count // len(seed) + 1))[:count]
    started = time.monotonic()
    response = request('/completion', {'prompt': tokens, 'n_predict': 128,
                       'temperature': 0, 'seed': 42, 'ignore_eos': True,
                       'cache_prompt': False}, timeout=M.get('full_context_timeout', 1800))
    save(dest/'full-context.json', response)
    result = {'input_tokens': count, 'seconds': time.monotonic()-started,
              'evaluated_tokens': response.get('tokens_evaluated'),
              'generated_tokens': response.get('tokens_predicted'),
              'context_limit_reached': response.get('truncated'), 'timings': response.get('timings', {})}
    # The server sets truncated=true at an exact context boundary, even when
    # every requested input/output token was processed with context shift off.
    result['passed'] = (result['evaluated_tokens'] == count
                        and result['generated_tokens'] == 128
                        and result['timings'].get('prompt_n') == count
                        and result['timings'].get('cache_n') == 0)
    return result

def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except sp.TimeoutExpired:
            proc.kill()
            proc.wait()

def fa_failure(path):
    result_path=path/'result.json'
    log_path=path/'server.log'
    if not result_path.exists() or not log_path.exists():
        return False
    result=json.loads(result_path.read_text())
    return (result.get('backend')=='hip' and result.get('fa')=='on'
            and result.get('status')=='failed'
            and 'GGML_ASSERT(max_blocks_per_sm > 0)' in log_path.read_text(errors='replace'))

if M.get('mode') == 'mtp-slots-cache':
    import mtp
    mtp.run()
    raise SystemExit(0)

cases=[]
for model in M['models']:
    for backend in M['backends']:
        for cache in M['caches']:
            original=f"{model['id']}-{backend}-{cache}"
            if M.get('retry_from'):
                if fa_failure(Path(M['retry_from'])/original):
                    cases.append((model,backend,cache,'off',original))
            else:
                cases.append((model,backend,cache,'on',None))

status_path=Path('status.json')
previous=json.loads(status_path.read_text()) if status_path.exists() else {}
state={'start_utc': previous.get('start_utc') or M.get('start_utc') or now(),
       'runner_start_utc':now(), 'pid':os.getpid(), 'status':'running',
       'completed':[], 'total':len(cases),
       'source_revision':sp.check_output(['git','-C',M['source'],'rev-parse','HEAD'],text=True).strip()}
save(status_path,state)
was_active=previous.get('service_was_active', sp.run(['systemctl','is-active','--quiet',M['service']]).returncode==0)
state['service_was_active']=was_active
save(status_path,state)
server=None
try:
    for model in M['models']:
        for field in ['file','mmproj']:
            if not Path(model[field]).is_file():
                raise FileNotFoundError(model[field])
    sp.run(['sudo','systemctl','stop',M['service']],check=True)
    for model,backend,cache,fa,fallback_of in cases:
        context = int(model.get('context', M.get('context', 16384)))
        key=f"{model['id']}-{backend}-{cache}"+('-fa-off' if fa=='off' else '')
        dest=ROOT/key
        dest.mkdir(exist_ok=True)
        result_path=dest/'result.json'
        if result_path.exists():
            if fa=='on' and fa_failure(dest):
                cases.append((model,backend,cache,'off',key))
                state['total']=len(cases)
            state['completed'].append(key)
            continue
        state['phase']='benchmark'
        state['current']=key
        save(status_path,state)
        result={'id':key,'model':model,'backend':backend,'cache':cache,
                'fa':fa,'fallback_of':fallback_of,'context':context,'benchmark_repetitions':3,'image_repetitions':1,'start_utc':now(),
                'images':[], 'errors':[]}
        binary=f"{M['source']}/build-{backend}/bin"
        estimate = model.get('estimated_vram_mib', {}).get(backend, {}).get(cache)
        result['estimated_vram_mib'] = estimate
        if estimate and estimate > M.get('vram_limit_mib', 16384):
            result.update(status='excluded', errors=['Estimated VRAM exceeds 16 GiB'])
            save(result_path, result)
            state['completed'].append(key)
            save(status_path, state)
            continue
        # Allocate the requested context without filling it in pp512/tg128.
        bench=[f'{binary}/llama-bench','-m',model['file'],'-p','512','-n','128',
               '-r','3','-t','10','-b','2048','-ub','512','-ngl','99',
               '-fa',fa,'-ctk',cache,'-ctv',cache,'-fitc',str(context),'-fitt','0',
               '-o','json','-v']
        partial = json.loads((dest/'partial.json').read_text()) if (dest/'partial.json').exists() else {}
        if partial.get('benchmark', {}).get('returncode') == 0:
            result['benchmark'] = partial['benchmark']
        else:
            result['benchmark']=command(bench,dest/'bench.json')
        if result['benchmark']['returncode'] != 0:
            result['errors'].append(f"benchmark exited {result['benchmark']['returncode']}")
        bench_log=(dest/'bench.json.stderr').read_text(errors='replace')
        result['benchmark']['placement']=placement(bench_log, context)
        try:
            result['benchmark']['results']=json.loads((dest/'bench.json').read_text())
        except (ValueError,OSError):
            result['errors'].append('benchmark failed or produced no JSON')
        save(dest/'partial.json',result)
        state['phase']='images'
        save(status_path,state)
        # One model load serves each of the four images once.
        args=[f'{binary}/llama-server','--model',model['file'],'--mmproj',model['mmproj'],
              '--alias',model['id'],'--ctx-size',str(context),'--parallel','1',
              '--flash-attn',fa,'--cache-type-k',cache,'--cache-type-v',cache,
              '--n-gpu-layers','all','--fit','off','--threads','10',
              '--batch-size',str(max(2048,model['ubatch'])),
              '--ubatch-size',str(model['ubatch']),
              '--image-min-tokens',str(model['image_min']),
              '--image-max-tokens',str(model['image_max']),
              '--host','127.0.0.1','--port',str(M['port']),
              '--jinja','--reasoning-budget','0','--log-verbosity','4']
        if M.get('full_context'):
            args += ['--no-context-shift', '--no-kv-unified']
        result['server_command']=args
        monitor = None
        try:
            with open(dest/'server.log','w') as log:
                server=sp.Popen(args,stdout=log,stderr=sp.STDOUT)
                monitor = MemoryMonitor(server, dest)
                deadline=time.monotonic()+300
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'server exited {server.returncode}')
                    try:
                        if request('/health',timeout=2).get('status')=='ok':
                            break
                    except (OSError,ValueError):
                        pass
                    if time.monotonic()>deadline:
                        raise TimeoutError('server health timeout')
                    time.sleep(2)
                result['props']=request('/props')
                for image in IMAGES:
                    encoded=base64.b64encode((ROOT/'media'/image['file']).read_bytes()).decode()
                    for repetition in range(1):
                        sample={'image':image['id'],'repetition':repetition+1,'seed':42+repetition}
                        payload={'model':model['id'],'messages':[{'role':'user','content':[
                            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}},
                            {'type':'text','text':image['prompt']}]}],
                            'temperature':0.2,'seed':sample['seed'],'max_tokens':512,
                            'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
                        started=time.monotonic()
                        try:
                            saved = [item for item in partial.get('images', []) if item.get('image') == image['id'] and item.get('repetition') == 1 and 'error' not in item]
                            if saved:
                                result['images'].append(saved[0])
                                continue
                            response=request('/v1/chat/completions',payload)
                            sample['seconds']=time.monotonic()-started
                            save(dest/f"{image['id']}-{repetition+1}.json",response)
                            choice=response['choices'][0]
                            sample.update({'text':choice['message'].get('content') or '',
                                           'finish_reason':choice['finish_reason'],
                                           'usage':response.get('usage',{}),
                                           'timings':response.get('timings',{})})
                            sample.update(score(sample['text'],image))
                            count=sample['usage'].get('completion_tokens',0)
                            sample['end_to_end_tokens_per_second']=count/sample['seconds']
                        except Exception as exc:
                            sample['error']=str(exc)
                            sample['seconds']=time.monotonic()-started
                        result['images'].append(sample)
                        save(dest/'partial.json',result)
                if M.get('full_context'):
                    state['phase'] = 'full_context'
                    save(status_path, state)
                    result['full_context'] = full_context(context, dest)
                    if not result['full_context']['passed']:
                        result['errors'].append('Full-context token counts do not match the request')
                    save(dest/'partial.json', result)
        except Exception as exc:
            result['errors'].append(str(exc))
        finally:
            result['server_exit_before_cleanup']=server.poll() if server else None
            if monitor:
                result['peak_memory_mib'] = monitor.finish()
            stop(server)
            server=None
        log=(dest/'server.log').read_text(errors='replace')
        result['server_placement']=placement(log, context)
        result['projector_backend']=re.findall(r'CLIP using ([^\n]+) backend',log)
        result['actual_image_tokens']=re.findall(r'image.*?(\d+) tokens',log)
        for kind, details in [('benchmark', result['benchmark']['placement']),
                              ('server', result['server_placement'])]:
            if not details['full_offload'] or not details['context_matches']:
                result['errors'].append(f'{kind}: full GPU offload or requested context not confirmed')
            if not details['flash_attention'] or any(('enabled' if fa=='on' else 'disabled') not in flag for flag in details['flash_attention']):
                result['errors'].append(f'{kind}: Flash Attention not confirmed')
        expected_backend = 'ROCm' if backend == 'hip' else 'Vulkan'
        if not any(expected_backend in value for value in result['projector_backend']):
            result['errors'].append('GPU projector placement not confirmed')
        result['stop_utc']=now()
        result['status']='complete' if len(result['images'])==4 and not result['errors'] and not any('error' in s for s in result['images']) else 'failed'
        if fa=='off' and 'quantized V cache requires flash_attn to be enabled' in log:
            result['status']='unsupported'
            result['errors']=['FA off is unsupported with quantized V cache; use f16/f16 for the fallback']
        save(result_path,result)
        if fa=='on' and fa_failure(dest):
            cases.append((model,backend,cache,'off',key))
            state['total']=len(cases)
        state['completed'].append(key)
        save(status_path,state)
    state['status']='finished'
except BaseException as exc:
    state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
    state['error']=str(exc)
    raise
finally:
    stop(server)
    if was_active:
        state['restore_returncode']=sp.run(['sudo','systemctl','start',M['service']]).returncode
    state['stop_utc']=now()
    state['total_seconds']=(dt.datetime.fromisoformat(state['stop_utc'])-dt.datetime.fromisoformat(state['start_utc'])).total_seconds()
    save(status_path,state)
