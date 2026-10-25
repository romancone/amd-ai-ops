#!/usr/bin/env python3
"""Render condensed evidence and one answer per image; raw logs remain external."""
import argparse
import hashlib
import json
import re
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('--patched-run',type=Path,action='append',default=[])
parser.add_argument('--additional-run',type=Path,action='append',default=[])
parser.add_argument('--reviews',type=Path,help='Luna reference-based judgements; original checklist reviews are not used')
args=parser.parse_args()
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
