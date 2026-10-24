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

def placement(log):
    layers = re.findall(r'offloaded (\d+)/(\d+) layers', log)
    ctx = re.findall(r'\bn_ctx\s*=\s*(\d+)', log)
    return {'layers': layers, 'context': ctx,
            'full_offload': bool(layers) and all(a == b for a,b in layers),
            'context_16k': bool(ctx) and all(int(n)==16384 for n in ctx),
            'flash_attention': re.findall(r'flash_attn\s*=\s*([^\n]+)', log)}

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
was_active=sp.run(['systemctl','is-active','--quiet',M['service']]).returncode==0
server=None
try:
    for model in M['models']:
        for field in ['file','mmproj']:
            if not Path(model[field]).is_file():
                raise FileNotFoundError(model[field])
    sp.run(['sudo','systemctl','stop',M['service']],check=True)
    for model,backend,cache,fa,fallback_of in cases:
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
        state['current']=key
        save(status_path,state)
        result={'id':key,'model':model,'backend':backend,'cache':cache,
                'fa':fa,'fallback_of':fallback_of,'context':16384,'benchmark_repetitions':3,'image_repetitions':1,'start_utc':now(),
                'images':[], 'errors':[]}
        binary=f"{M['source']}/build-{backend}/bin"
        # fit minimum fixes allocation at 16K without filling the cache.
        bench=[f'{binary}/llama-bench','-m',model['file'],'-p','512','-n','128',
               '-r','3','-t','10','-b','2048','-ub','512','-ngl','99',
               '-fa',fa,'-ctk',cache,'-ctv',cache,'-fitc','16384','-fitt','0',
               '-o','json','-v']
        partial = json.loads((dest/'partial.json').read_text()) if (dest/'partial.json').exists() else {}
        if partial.get('benchmark', {}).get('returncode') == 0:
            result['benchmark'] = partial['benchmark']
        else:
            result['benchmark']=command(bench,dest/'bench.json')
        if result['benchmark']['returncode'] != 0:
            result['errors'].append(f"benchmark exited {result['benchmark']['returncode']}")
        bench_log=(dest/'bench.json.stderr').read_text(errors='replace')
        result['benchmark']['placement']=placement(bench_log)
        try:
            result['benchmark']['results']=json.loads((dest/'bench.json').read_text())
        except (ValueError,OSError):
            result['errors'].append('benchmark failed or produced no JSON')
        save(dest/'partial.json',result)
        # One model load serves each of the four images once.
        args=[f'{binary}/llama-server','--model',model['file'],'--mmproj',model['mmproj'],
              '--alias',model['id'],'--ctx-size','16384','--parallel','1',
              '--flash-attn',fa,'--cache-type-k',cache,'--cache-type-v',cache,
              '--n-gpu-layers','all','--fit','off','--threads','10',
              '--batch-size',str(max(2048,model['ubatch'])),
              '--ubatch-size',str(model['ubatch']),
              '--image-min-tokens',str(model['image_min']),
              '--image-max-tokens',str(model['image_max']),
              '--host','127.0.0.1','--port',str(M['port']),
              '--jinja','--reasoning-budget','0','--log-verbosity','4']
        result['server_command']=args
        try:
            with open(dest/'server.log','w') as log:
                server=sp.Popen(args,stdout=log,stderr=sp.STDOUT)
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
        except Exception as exc:
            result['errors'].append(str(exc))
        finally:
            result['server_exit_before_cleanup']=server.poll() if server else None
            stop(server)
            server=None
        log=(dest/'server.log').read_text(errors='replace')
        result['server_placement']=placement(log)
        result['projector_backend']=re.findall(r'CLIP using ([^\n]+) backend',log)
        result['actual_image_tokens']=re.findall(r'image.*?(\d+) tokens',log)
        for kind, details in [('benchmark', result['benchmark']['placement']),
                              ('server', result['server_placement'])]:
            if not details['full_offload'] or not details['context_16k']:
                result['errors'].append(f'{kind}: full GPU offload or 16K context not confirmed')
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
