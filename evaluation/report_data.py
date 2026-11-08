"""Shared measurements, configurations and image answers for HTML reports."""
import json
import re
import hashlib

PREFIX = 'window.BENCHMARK_DATA = '


MODEL_IDS = {
    'Gemma 12B Q4_0': 'gemma12-Q4_0',
    'Gemma E4B Q4_0': 'gemma-e4b-Q4_0',
    'Qwen 9B Q4_K_M': 'qwen35-MTP-Q4_K_M',
    'Qwen 9B MTP Q4_K_M': 'qwen35-MTP-Q4_K_M',
}
CACHED_GROUPS = ('cases','reruns','context64','two_user_cases','context256','extension_cases')


def model_id(value):
    if isinstance(value,dict):
        value=value['id']
    return MODEL_IDS.get(value, value)


def archive_table(data, block, rows):
    if not rows:
        return
    saved = {k:v for k,v in block.items() if k != 'rows'}
    saved['rows'] = rows
    attempt = hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()[:20]
    data.setdefault('views', {}).setdefault('history', {})[attempt] = saved


def load_data(repo):
    path = repo / 'reports/benchmark-data.js'
    if not path.exists():
        return {}
    data = json.loads(path.read_text()[len(PREFIX):].rstrip().removesuffix(';'))
    return expand_store(data) if data.get('schema_version') == 3 else data


def save_data(repo, data):
    data['identities'] = {'models': MODEL_IDS}
    def encode(value, depth=0):
        compact = lambda v: json.dumps(v, ensure_ascii=False)
        if isinstance(value, dict):
            if depth >= 2:
                return compact(value)
            items = [compact(k) + ': ' + encode(v, depth + 1) for k, v in value.items()]
            opening, closing = '{', '}'
        elif isinstance(value, list):
            if not value or all(not isinstance(v, dict) for v in value):
                return compact(value)
            items = [encode(v, depth + 1) for v in value]
            opening, closing = '[', ']'
        else:
            return compact(value)
        return opening + '\n' + ',\n'.join('  ' * (depth + 1) + v for v in items) + '\n' + '  ' * depth + closing
    encoded = encode(split_store(data)).replace('<', '\\u003c')
    (repo / 'reports/benchmark-data.js').write_text(PREFIX + encoded + ';\n')
    write_benchmark(repo, data)


def update_data(repo, key, value):
    data = load_data(repo)
    if key == 'models':
        previous = data.get(key, {})
        rows, answers, answer_ids = {}, [], {}
        for collection in (previous, value):
            for original in collection.get('rows', []):
                if (original.get('context') != 16384 or original.get('status') == 'excluded'
                        or original.get('family') == 'qwen35-mtp'):
                    continue
                row = json.loads(json.dumps(original))
                for sample in row.get('images', {}).values():
                    if 'answer_id' not in sample:
                        continue
                    answer = collection['answers'][sample['answer_id']]
                    digest = json.dumps(answer,sort_keys=True,ensure_ascii=False)
                    if digest not in answer_ids:
                        answer_ids[digest] = len(answers)
                        answers.append(answer)
                    sample['answer_id'] = answer_ids[digest]
                identity = tuple(row[k] for k in ('model','backend','cache','context','fa'))
                # An absent run file must not replace a recorded result with pending.
                if row.get('status') != 'pending' or identity not in rows:
                    rows[identity] = row
        merged = {**previous, **value, 'rows':list(rows.values()), 'answers':answers}
        if previous and previous != merged:
            digest=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()[:20]
            data.setdefault('models_history',{})[digest]=previous
        value = merged
    data[key] = value
    save_data(repo, data)


def merge_short_results(repo, results, rows, answers):
    data = load_data(repo)
    blocks = data['views']['mtp']
    candidates = [block for block in blocks.values() if block['kind'] == 'rows' and block['rows']
                  and all(len(row['cells']) == 16 and row['cells'][2]['value'] == '1'
                          and row['cells'][3]['value'] == 'on' for row in block['rows'])]
    if not candidates:
        raise ValueError('Cannot identify the single-user MTP-on table')
    key = lambda row: tuple(cell['value'] for cell in row['cells'][:5])
    combined = {key(row):row for block in candidates for row in block['rows']}
    table = candidates[0]['rows']
    table[:] = combined.values()
    for block in candidates[1:]:
        block['rows'] = []
    incoming = {key(row): row for row in rows}
    known = {key(row) for row in table}
    archive_table(data,candidates[0],[row for row in table if key(row) in incoming and row != incoming[key(row)]])
    table[:] = [incoming.get(key(row), row) for row in table]
    table.extend(row for identity, row in incoming.items() if identity not in known)
    existing = data.setdefault('short', {}).get('cases', {})
    metadata={k:v for k,v in data['short'].items() if k!='cases'}
    if metadata:
        metadata['case_digests']={k:v['result_sha256'] for k,v in existing.items()}
        digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:20]
        data.setdefault('short_imports',{})[digest]=metadata
    for key,case in results['cases'].items():
        if key in existing and existing[key] != case:
            digest=hashlib.sha256(json.dumps(existing[key],sort_keys=True).encode()).hexdigest()[:20]
            data.setdefault('short_history',{})[digest]=existing[key]
    existing.update(results['cases'])
    data['short'].update(results, cases=existing)
    digest = hashlib.sha256(json.dumps(answers,sort_keys=True).encode()).hexdigest()[:16]
    blocks['short-answers:'+digest] = {'kind': 'answer', 'answers': answers}
    save_data(repo, data)

def publish_section(repo, name, records, source_run):
    """Merge structured measurements and saved answers by configuration."""
    data = load_data(repo)
    blocks = data.setdefault('views', {}).setdefault('mtp', {})
    incoming = {f'{name}:{i}':dict(block,source=name) for i,block in enumerate(records)}
    run_id=hashlib.sha256(str(source_run).encode()).hexdigest()[:20]
    for block in incoming.values():
        if block['kind']=='rows':
            for row in block['rows']:
                row['attempt_id']=hashlib.sha256((run_id+json.dumps(row,sort_keys=True)).encode()).hexdigest()[:20]
    for key, block in incoming.items():
        if block['kind'] != 'rows':
            digest = hashlib.sha256(json.dumps(block['answers'],sort_keys=True).encode()).hexdigest()[:16]
            blocks['answer:'+digest] = block
            continue
        headers = block['headers']
        config_columns = [i for i,h in enumerate(headers) if h in (
            'Model','Model / backend','Backend','Users','MTP','K/V cache','Cache','Cache K/V','FA',
            'Context/user','Allocated / usable per user','Total allocated / usable','Allocated tokens')]
        identity = lambda row: tuple(model_id(row['cells'][i]['value']) if headers[i]=='Model' else row['cells'][i]['value'] for i in config_columns) if config_columns else tuple(c['value'] for c in row['cells'])
        incoming_rows = {identity(row):row for row in block['rows']}
        remaining = dict(incoming_rows)
        for old in list(blocks.values()):
            if old['kind'] != 'rows' or old.get('headers') != headers:
                continue
            archive_table(data,old,[row for row in old['rows'] if identity(row) in incoming_rows and row != incoming_rows[identity(row)]])
            old['rows'] = [incoming_rows.get(identity(row),row) for row in old['rows']]
            for row in old['rows']:
                remaining.pop(identity(row),None)
        if remaining:
            block['rows'] = list(remaining.values())
            # Content keys keep unrelated imported tables from being overwritten.
            digest = hashlib.sha256(json.dumps(block,sort_keys=True).encode()).hexdigest()[:12]
            blocks[f'{name}:{digest}'] = block
    save_data(repo, data)


def make_table(headers, values, title):
    """Keep column values structured; retain auxiliary memory readings separately."""
    keep = [i for i,name in enumerate(headers) if 'GPU' in name or not re.search(r'\b(?:RAM|GTT)\b',name)]
    combined = {i for i in keep if ' / RAM / GTT' in headers[i]}
    out = dict(kind='rows',title=title,headers=[headers[i].replace(' / RAM / GTT','') for i in keep],rows=[])
    for row in values:
        value = lambda v: f'{v:.2f}' if isinstance(v,float) else '—' if v is None else str(v)
        out['rows'].append(dict(cells=[dict(value=value(row[i]).split(' / ',1)[0] if i in combined else value(row[i])) for i in keep],
                               supplemental={headers[i]:row[i] for i in range(len(headers)) if i not in keep or i in combined}))
    return out


def refresh_comparison(repo, cached):
    """Refresh matching widget timings without changing selected profiles or scores."""
    data = load_data(repo)
    limits = {'gemma12-Q4_0':262144, 'gemma-e4b-Q4_0':131072, 'qwen35-MTP-Q4_K_M':262144}
    for group in CACHED_GROUPS:
        for case in cached.get(group, []):
            limit = limits.get(model_id(case['model']))
            if limit is None or not 0 < case['context_per_user'] <= limit:
                raise ValueError('Unknown model or context above the model limit: '+str(case['model']))
            if case.get('status') == 'passed' and (
                    case.get('new_input_tokens_per_user') != 512 or case.get('output_tokens_per_user') != 128
                    or not isinstance(case.get('cached_input_tokens_per_user'),(int,float))
                    or case['cached_input_tokens_per_user'] <= 0
                    or not isinstance(case.get('ttft_seconds_mean'),(int,float))
                    or not isinstance(case.get('decode_aggregate_tps'),(int,float))):
                raise ValueError('Long results require cached history, 512 new tokens, TTFT and decode-only throughput')
    previous = data.get('cached', {})
    retained = data.setdefault('cached_history', {})
    for group in CACHED_GROUPS:
        incoming = {case.get('result_sha256') or hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest() for case in cached.get(group,[])}
        for case in previous.get(group,[]):
            digest = case.get('result_sha256') or hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
            if digest not in incoming:
                retained[digest] = case
    merged = dict(previous, **cached)
    for group in CACHED_GROUPS:
        records = {}
        for case in previous.get(group, []) + cached.get(group, []):
            digest = case.get('result_sha256') or hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
            records[digest] = case
        merged[group] = list(records.values())
    cached = merged
    data['cached'] = cached
    latest = {}
    for group in ('cases', 'reruns', 'context64', 'two_user_cases', 'context256', 'extension_cases'):
        for case in cached.get(group, []):
            key = (model_id(case['model']), case['backend'].lower(), case['slots'],
                   case['cache'], case['context_per_user'], case['mtp'])
            latest[key] = case
    for profile in data.get('comparison', {}).get('profiles', []):
        for case in latest.values():
            if case['status'] == 'pending' or case['slots'] != profile['u'] or case['context_per_user'] != int(profile['c'][:-1]) * 1024:
                continue
            if profile['c'] == '16K' and profile['u'] != 4:
                continue
            name = next((name for name, identity in MODEL_IDS.items() if identity == model_id(case['model'])), case['model'])
            backend = {'hip': 'HIP', 'vulkan': 'Vulkan'}[case['backend'].lower()]
            cache = {'q4_0': 'Q4/Q4', 'q8_0': 'Q8/Q8', 'f16': 'f16/f16'}[case['cache']]
            if not any(model_id(row[0]) == model_id(name) and row[1] == backend and row[2] == cache for row in profile['rows']):
                profile['rows'].append([name, backend, cache] + [['Not measured', 'Not measured'] for _ in range(5)] + [{}])
        for row in profile['rows']:
            cache = {'Q4/Q4': 'q4_0', 'Q8/Q8': 'q8_0', 'f16/f16': 'f16'}[row[2]]
            changed = False
            for state in (0, 1):
                key = (model_id(row[0]), row[1].lower(), profile['u'], cache,
                       int(profile['c'][:-1]) * 1024, bool(state))
                case = latest.get(key)
                if not case or case['status'] == 'pending':
                    continue
                changed = True
                status = {'oom': 'OOM', 'timeout': 'Timeout'}.get(case['status'], 'Fail')
                for column, field in ((3, 'ttft_seconds_mean'), (4, 'prompt_tps_per_user_mean'), (5, 'decode_aggregate_tps')):
                    if case['status'] == 'passed':
                        if isinstance(case.get(field),(int,float)):
                            row[column][state] = round(case[field],2)
                        elif not isinstance(row[column][state],(int,float)):
                            row[column][state] = 'Not measured'
                    else:
                        row[column][state] = 'Fail' if row[column][state]=='Fail' else status
                peak = case.get('peak_gpu_mib')
                row[7][state] = (round(peak / 1024, 2) if case['status'] == 'passed' and peak is not None
                                 else f'{peak / 1024:.2f} before timeout' if case['status'] == 'timeout' and peak is not None
                                 else 'Not measured' if case['status']=='passed' else 'Fail' if row[7][state]=='Fail' else status)
            if changed and len(row) > 8:
                row[8] = {}
    save_data(repo, data)


def expand_store(store):
    def expand(value):
        if isinstance(value, list):
            return [expand(v) for v in value]
        if not isinstance(value, dict):
            return value
        if '$merge' in value:
            return {k:v for item in value['$merge'] for k,v in expand(item).items()}
        if value.get('kind')=='rows' and 'columns' in value:
            result={k:expand(v) for k,v in value.items() if k not in ('columns','rows','row_attributes','cell_overrides')}
            result['rows']=[]
            for i, record_id in enumerate(value['rows']):
                record=expand(store['measurements'][record_id])
                cells=[]
                for j,column in enumerate(value['columns']):
                    cells.append(dict(value=record[column['source']][column['name']]))
                result['rows'].append(dict(cells=cells,attempt_id=record.get('attempt_id',record_id),
                                           **({'supplemental':record['supplemental']} if record.get('supplemental') else {})))
            return result
        if '$ref' in value:
            result = store[value['$ref']][value['id']]
            for key in value.get('path', []):
                result = result[key]
            result = expand(result)
            if 'split' in value:
                result = result.split(value['split'][0])[value['split'][1]]
            if 'offset' in value:
                result += value['offset']
            if value.get('commas'):
                result = f'{int(result):,}'
            if 'scale' in value:
                result = round(float(result) * value['scale'], 2)
            if 'fixed' in value:
                result = f"{float(result):.{value['fixed']}f}"
            return result
        return {k: expand(v) for k, v in value.items()}
    return expand(store['catalog'])


def split_store(data):
    """Store measurements once; view selections contain references, never copies."""
    import copy
    catalog = copy.deepcopy(data)
    store = dict(schema_version=3, configurations={}, measurements={}, image_answers={})
    def intern(section, value):
        key = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        store[section][key] = value
        return {'$ref': section, 'id': key}
    def field(ref, *path, scale=None):
        result = dict(ref, path=list(path))
        if scale is not None:
            result['scale'] = scale
        return result
    def answer(value):
        if isinstance(value, dict):
            value = dict(value)
            if isinstance(value.get('text'),str):
                value['text'] = intern('image_answers',value['text'])
        return intern('image_answers',value)

    candidates = []
    capacity_sources = []
    aliases = {'gemma12-Q4_0':'Gemma 12B Q4_0', 'gemma-e4b-Q4_0':'Gemma E4B Q4_0',
               'qwen35-MTP-Q4_K_M':'Qwen 9B Q4_K_M', 'Qwen 9B MTP Q4_K_M':'Qwen 9B Q4_K_M'}
    cache = lambda v: v.replace('q4_0','Q4').replace('q8_0','Q8')
    short_sources = []
    def pack_short(case, active):
        configuration = {k:v for k,v in case.items() if k not in ('off','on','counterpart','result_path','result_sha256','restoration')}
        conf = intern('configurations',configuration)
        for k in configuration:
            case[k]=field(conf,k)
        for state in ('off','on'):
            for i, image_answer in enumerate(case[state].get('images', [])):
                case[state]['images'][i]=answer(image_answer)
            raw = copy.deepcopy(case[state])
            record=intern('measurements',dict(configuration=conf,workload='short',mtp=state,metrics=case[state]))
            if active:
                short_sources.append((configuration,state,raw,record))
            case[state]=field(record,'metrics')
    for case in catalog.get('short',{}).get('cases',{}).values():
        pack_short(case,True)
    for case in catalog.get('short_history',{}).values():
        pack_short(case,False)
    cached_sources = []
    for group in ('cases','reruns','context64','two_user_cases','context256','extension_cases'):
        for i, case in enumerate(catalog.get('cached', {}).get(group, [])):
            conf = intern('configurations', {k:case[k] for k in ('model','backend','slots','cache','mtp','context_per_user')})
            metrics = {k:v for k,v in case.items() if k not in ('model','backend','slots','cache','mtp','context_per_user')}
            record = intern('measurements',dict(configuration=conf,workload='cached-continuation',metrics=metrics))
            cached_sources.append((case,record))
            catalog['cached'][group][i] = {'$merge':[conf,field(record,'metrics')]}
    for key,case in catalog.get('cached_history',{}).items():
        conf=intern('configurations',{k:case[k] for k in ('model','backend','slots','cache','mtp','context_per_user')})
        metrics={k:v for k,v in case.items() if k not in ('model','backend','slots','cache','mtp','context_per_user')}
        record=intern('measurements',dict(configuration=conf,workload='cached-continuation',metrics=metrics))
        catalog['cached_history'][key]={'$merge':[conf,field(record,'metrics')]}
    if 'repeats' in catalog:
        conf=intern('configurations',catalog['repeats']['settings'])
        metrics={k:v for k,v in catalog['repeats'].items() if k!='settings'}
        record=intern('measurements',dict(configuration=conf,workload='image-quality-repeats',metrics=metrics))
        catalog['repeats']={'settings':conf,**{k:field(record,'metrics',k) for k in metrics}}
    # Table schemas remain presentation metadata. Cell values live in records.
    config_headers = {'Model','Backend','Users','MTP','FA','K/V cache','Cache','Cache K/V'}
    for blocks in catalog.get('views', {}).values():
        for block in blocks.values():
            if block['kind'] == 'answer':
                block['answers'] = [answer(v) for v in block['answers']]
                continue
            headers = block.get('headers', [])
            for row in block['rows']:
                cells = row['cells']
                values = [c['value'] for c in cells]
                names = headers if len(headers) == len(cells) else [str(i) for i in range(len(cells))]
                config = {name: values[i] for i, name in enumerate(names) if name in config_headers}
                conf = intern('configurations', config)
                metrics = {name: values[i] for i, name in enumerate(names) if name not in config_headers}
                if len(values)==16 and values[0] in aliases:
                    source = next(((raw,record) for cfg,state,raw,record in short_sources
                                   if model_id(cfg['model'])==model_id(values[0]) and cfg['backend'].lower()==values[1].lower()
                                   and str(cfg['users'])==values[2] and state==values[3]
                                   and cache(cfg['cache']+'/'+cfg['cache'])==cache(values[4])),None)
                    if source:
                        for idx,metric,scale in [(6,'prefill_tps_mean',1),(7,'decode_tps_mean',1),(10,'ttft_seconds_mean',1000),(11,'request_seconds_mean',1),(12,'draft_acceptance_percent',1),(13,'image_quality_percent',1),(14,'peak_gpu_mib',1/1024)]:
                            raw=source[0].get(metric)
                            try:
                                same=isinstance(raw,(int,float)) and round(raw*scale,2)==float(values[idx])
                            except ValueError:
                                same=False
                            if same:
                                digits=len(values[idx].split('.')[1]) if '.' in values[idx] else 0
                                metrics[names[idx]]=dict(field(source[1],'metrics',metric,scale=scale),fixed=digits)
                workload = 'llama-bench' if 'pp512 ± SD' in names else 'short' if len(values)==16 and 'Draft accept %' in names else block.get('source','recorded')
                attempt=row.get('attempt_id') or hashlib.sha256(json.dumps(values).encode()).hexdigest()[:20]
                record = intern('measurements', dict(configuration=conf, workload=workload, attempt_id=attempt, metrics=metrics,
                                                      **({'supplemental':row['supplemental']} if row.get('supplemental') else {})))
                row['_measurement']=record['id']
                for i, cell in enumerate(cells):
                    cell['value'] = field(conf, names[i]) if names[i] in config_headers else field(record, 'metrics', names[i])
                if len(values)==10 and values[1] in ('HIP','Vulkan') and ' / ' in values[5] and ' / ' in values[6]:
                    capacity_sources.append((values,record,names))
                if len(values) == 16 and values[0] in aliases and values[3] in ('off','on'):
                    candidates.append((aliases[values[0]], values[1].lower(), cache(values[4]), int(values[2]), values[3], record, names, values))
    for blocks in catalog.get('views', {}).values():
        for block in blocks.values():
            if block['kind']!='rows' or not block['rows']:
                continue
            columns=[]
            for cell in block['rows'][0]['cells']:
                ref=cell['value']
                columns.append(dict(name=ref['path'][-1],source='configuration' if ref['$ref']=='configurations' else 'metrics'))
            block.update(columns=columns, rows=[row['_measurement'] for row in block['rows']])
    for model_data in [catalog.get('models',{}),*catalog.get('models_history',{}).values()]:
        for row_index,row in enumerate(model_data.get('rows', [])):

            config_keys = ('model','backend','cache','fa','context','patch')
            config = {k: row[k] for k in config_keys if k in row}
            ref = intern('configurations', config)
            for k in config:
                row[k] = field(ref, k)
            metrics = {k:v for k,v in row.items() if k not in config_keys}
            measurement = intern('measurements', dict(configuration=ref, workload='llama-bench-and-images', metrics=metrics))
            model_data['rows'][row_index]={'$merge':[ref,field(measurement,'metrics')]}
        answers = model_data.get('answers', [])
        for i, image_answer in enumerate(answers):
            answers[i] = answer(image_answer)
    # Preserve manually chosen scores/timings; reference their source when equal.
    for profile in catalog.get('comparison', {}).get('profiles', []):
        for row in profile['rows']:
            config = dict(model=row[0],backend=row[1],cache=row[2],users=profile['u'],context=profile['c'],
                          kv='Unified' if profile['c']=='16K' and profile['u']!=4 else 'Separate')
            conf = intern('configurations', config)
            for state, label in enumerate(('off','on')):
                fields = {name: row[i][state] for i,name in enumerate(('ttft_s','prefill_tps_user','decode_tps_total','quality_percent','gpu_gib'),3)}
                refs = {}
                if config['kv']=='Unified':
                    source = next((c for c in candidates if c[:5] == (row[0],row[1].lower(),row[2],profile['u'],label)),None)
                    if source:
                        for name, idx, scale in [('ttft_s',10,.001),('prefill_tps_user',6,1),('decode_tps_total',7,profile['u']),('quality_percent',13,1),('gpu_gib',14,1)]:
                            try:
                                same = isinstance(fields[name], (int,float)) and round(float(source[7][idx])*scale,2)==fields[name]
                            except ValueError:
                                same = False
                            if same:
                                refs[name] = field(source[5], 'metrics', source[6][idx], scale=scale)
                if config['kv']=='Separate':
                    source = next(((c,r) for c,r in reversed(cached_sources)
                                   if aliases.get(c['model'],c['model'])==row[0] and c['backend'].lower()==row[1].lower()
                                   and c['slots']==profile['u'] and c['context_per_user']==int(profile['c'][:-1])*1024
                                   and cache(c['cache']+'/'+c['cache'])==row[2] and c['mtp']==bool(state)),None)
                    if source:
                        for name, metric, scale in [('ttft_s','ttft_seconds_mean',1),('prefill_tps_user','prompt_tps_per_user_mean',1),('decode_tps_total','decode_aggregate_tps',1),('gpu_gib','peak_gpu_mib',1/1024)]:
                            if source[0].get('status')=='passed' and isinstance(fields[name],(int,float)) and metric in source[0] and round(source[0][metric]*scale,2)==fields[name]:
                                refs[name]=field(source[1],'metrics',metric,scale=scale)
                if 'quality_percent' not in refs:
                    source=next((c for c in candidates if c[:5] == (row[0],row[1].lower(),row[2],profile['u'],label)
                                 and re.fullmatch(r'\d+(?:\.\d+)?',c[7][13]) and float(c[7][13])==fields['quality_percent']),None)
                    if source:
                        refs['quality_percent']=field(source[5],'metrics',source[6][13],scale=1)
                remainder = {k:v for k,v in fields.items() if k not in refs}
                if remainder:
                    record = intern('measurements', dict(configuration=conf, workload='selected-short' if config['kv']=='Unified' else 'cached-continuation', mtp=label, metrics=remainder))
                    refs.update({k:field(record,'metrics',k) for k in remainder})
                for i,name in enumerate(fields,3):
                    row[i][state] = refs[name]
            for i,k in enumerate(('model','backend','cache')):
                row[i] = field(conf,k)
            profile.pop('note',None)
    for users, rows in catalog.get('comparison', {}).get('contexts', {}).items():
        for i,row in enumerate(rows):
            conf = intern('configurations',dict(model=row[0],backend=row[2],cache=row[3],mtp=row[4],users=int(users),kv=row[6]))
            selected = list(row)
            for values,record,names in capacity_sources:
                slots = int(values[6].split(' / ')[0].replace(',','')) / int(values[5].split(' / ')[0].replace(',',''))
                if (aliases.get(values[0],values[0])==row[0] and values[1] in row[2].split(' / ')
                    and cache(values[2].split(' (')[0])==row[3] and values[4]==row[4] and slots==int(users)
                    and values[9]=='Pass' and values[5].split(' / ')[1]==row[5]
                    and ('Separate' if 'separate' in values[2] else 'Unified')==row[6]):
                    selected[5]=dict(field(record,'metrics',names[5]),split=[' / ',1])
                    break
            if row[6]=='Separate':
                for case,record in reversed(cached_sources):
                    if (model_id(case['model'])==model_id(row[0]) and case['backend'].lower()==row[2].lower() and case['slots']==int(users)
                        and cache(case['cache']+'/'+case['cache'])==row[3] and case['mtp']==(row[4]=='on')
                        and case['status']=='passed' and f"{case['context_per_user']-(4 if case['mtp'] else 0):,}"==row[5]):
                        source_conf = store['measurements'][record['id']]['configuration']
                        selected[5]=dict(field(source_conf,'context_per_user'),offset=-4 if case['mtp'] else 0,commas=True)
                        break
            for column,key in [(0,'model'),(2,'backend'),(3,'cache'),(4,'mtp'),(6,'kv')]:
                selected[column]=field(conf,key)
            rows[i]=selected
    store['catalog'] = catalog
    return store


def write_benchmark(repo, data=None):
    """Export recorded results; never infer measurements or recommendations."""
    data = load_data(repo) if data is None else data
    number = lambda v: f'{v:.2f}' if isinstance(v,float) else 'Not recorded' if v is None else str(v)
    cell = lambda v: number(v).replace('|', '\\|').replace('\n','<br>')
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                         ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])
    out = ['# Benchmarks',
           'Generated from [the shared dataset](reports/benchmark-data.js). Do not edit this file by hand.',
           '[Model benchmarks with real prompts](reports/index.html) · [Model benchmarks with llama-bench](reports/llama-bench.html) · [Method](evaluation/README.md)',
           'TTFT is time to first token. Decode speed excludes prefill. End-to-end speed includes it. Image quality covers four saved descriptions, not overall model accuracy.',
           '## 1. Test settings', data.get('documentation',{}).get('hardware',''),
           table(['Setting','Value'],data.get('documentation',{}).get('settings',[])),
           '## 2. Model benchmarks with llama-bench']
    models=data.get('models',{})
    rows=[]
    for r in models.get('rows',[]):
        scores=[r['images'].get(i['id'],{}).get('quality') for i in models['rubric']]
        quality=sum(scores)/len(scores) if scores and all(isinstance(v,(int,float)) for v in scores) else None
        speed=lambda key: number(r.get(key))+' ± '+number(r.get(key+'_sd'))
        rows.append([r['model'],r['backend'],r['cache']+'/'+r['cache'],r['fa'],speed('pp512'),speed('tg128'),quality,
                     r['memory_mib']/1024 if r.get('memory_mib') is not None else None,r['status']])
    out.append(table(['Model','Backend','K/V cache','FA','pp512 t/s ± SD','tg128 t/s ± SD','Image quality %','GPU GiB','Status'],rows))
    out += ['## 3. Short prompts, MTP and capacity',
            'Short prompts use 16K per user. Capacity is a separate full-fill check; it does not measure long-context image quality. Retained outcomes are not silently replaced with passes.']
    for i,block in enumerate((b for b in data.get('views',{}).get('mtp',{}).values() if b['kind']=='rows' and b['rows']),1):
        title = re.sub(r'^\d+(?:\.\d+)*\s+', '', block.get('title','Recorded measurements'))
        out += [f"### 3.{i} {title}",
                table(block['headers'],[[c['value'] for c in r['cells']] for r in block['rows']])]
    out += ['## 4. Long prompts',
            'History is cached before adding 512 new tokens and generating 128 tokens. TTFT and decode refer to that continuation. Failures retain their status; failed fills have no continuation timing.']
    cached=data.get('cached',{})
    seen=set();rows=[]
    for group in CACHED_GROUPS:
        for r in cached.get(group,[]):
            digest=r.get('result_sha256') or hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            passed=r.get('status')=='passed'
            rows.append([model_id(r['model']),r['backend'],r['slots'],r['context_per_user'],r['cache'],
                         'on' if r['mtp'] else 'off', 'on' if r.get('kv_unified') else 'off',r['status'],
                         r.get('ttft_seconds_mean') if passed else None,
                         r.get('prompt_tps_per_user_mean') if passed else None,
                         r.get('decode_aggregate_tps') if passed else None,
                         r.get('peak_gpu_mib',0)/1024 if passed and r.get('peak_gpu_mib') is not None else None,group])
    out.append(table(['Model','Backend','Users','Context/user','K/V cache','MTP','Unified KV','Status','TTFT s mean','Prefill t/s/user','Decode t/s total','GPU GiB','Attempt group'],rows))
    out += ['## 5. Repeated image checks',
            'Separate repeated image checks; these do not replace the single-answer scores above.']
    def flatten(value,path=''):
        if isinstance(value,dict):
            return [row for k,v in value.items() for row in flatten(v, path+'.'+k if path else k)]
        return [[path,json.dumps(value,ensure_ascii=False) if isinstance(value,list) else value]]
    out.append(table(['Measurement','Value'],flatten(data.get('repeats',{}))))
    out += ['## 6. Saved image answers',
            'All answers, individual scores and Luna-6 references are available in the two linked reports and the shared dataset. Identical text is stored once; configuration links and reviews are retained.']
    strata = data.get('strata')
    if strata:
        out += ['## 7. Strata coding benchmark',
                '[Strata model benchmarks](reports/strata.html)', strata['hardware']]
        published = [r for r in strata['results'] if r.get('status') == 'verified']
        if published:
            short_runs = published.copy()
            q4 = strata.get('iq3_coding_kv')
            if q4:
                index = next((i + 1 for i, r in enumerate(short_runs) if r['quant'] == q4['quant']), len(short_runs))
                short_runs.insert(index, q4)
            out += ['Short coding prompt, one active request per run. TTFT is the wait for the first token. Decode includes reasoning and answer tokens and excludes prompt processing.',
                    table(['Quant','KV cache','Experts','TTFT s','Answer starts s','Decode t/s','RAM GiB','GPU GiB','Program'],
                          [[r['quant'],r.get('kv', strata['settings']['kv']),r['placement'],r['generation']['ttft_seconds'],
                            r['generation']['first_content_seconds'],r['generation']['tokens_per_second'],
                            r['peak_ram_bytes']/2**30,r['peak_gpu_bytes']/2**30 if r.get('peak_gpu_bytes') is not None else None,
                            'Verified repair' if r['program_status'] == 'repaired' else 'Verified original']
                           for r in short_runs]),
                    strata.get('validation_note',''),
                    'IQ4 required SSD expert reads and is included as a speed reference.']
            coding = strata.get('iq3_long_coding')
            if coding:
                out += [f"**Near-256K coding check.** {coding['quant']}, one slot, the same {coding['input_tokens']:,}-token input for each cache. Results are for the original programs.",
                        table(['KV cache','Result','Finding'], [[r['kv'],r['verdict'],r['finding']] for r in coding['results']]),
                        '[Two-user capacity and serving profile](reports/strata.html#two-user-capacity)']
        else:
            out.append('Corrected evaluation pending; earlier runs are diagnostic.')
    (repo/'BENCHMARK.md').write_text('\n\n'.join(v for v in out if v)+'\n')
