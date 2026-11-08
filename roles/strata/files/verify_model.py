#!/usr/bin/env python3
"""Verify prepared model shards against the pinned source manifest."""
import ast
import hashlib
import json
from pathlib import Path


def verify(config_path, source_path, quant):
    config = json.loads(Path(config_path).read_text())
    args = config['args']
    pins = None
    manifest = 'UNSLOTH_IQ4_XS_SHARDS' if quant == 'UD-IQ4_XS' else 'ISTA_FILES'
    for node in ast.parse(Path(source_path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == manifest for t in node.targets):
            pins = ast.literal_eval(node.value)
            break
    if not pins:
        raise ValueError(f'Pinned {manifest} manifest is missing')
    native = Path(args[args.index('--native') + 1])
    if f'{quant}-' not in native.name:
        raise ValueError(f'Expected {quant}, got {native.name}')
    if not config.get('model_name', '').lower().endswith(quant.lower()):
        raise ValueError('API model identity does not match the quantization')
    verified = []
    hashed_count = 0
    paths = ([native.parent / name for name in pins] if quant == 'UD-IQ4_XS'
             else [Path(args[args.index(flag) + 1]) for flag in ('--native', '--ple-gguf')])
    for path in paths:
        size, expected = pins[path.name]
        if path.stat().st_size != size:
            raise ValueError(f'Wrong shard size: {path}')
        marker = path.with_name(path.name + '.done')
        marked = marker.read_text() if marker.exists() else ''
        method = 'upstream_sha256_marker'
        current = marker.exists() and marker.stat().st_mtime_ns >= path.stat().st_mtime_ns
        if f'sha256 {expected}' not in marked or not current:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch: {path}')
            temp = marker.with_name(marker.name + '.tmp')
            temp.write_text(f'sha256 {expected}')
            temp.replace(marker)
            method = 'hashed_now'
            hashed_count += 1
        verified.append({'path': str(path), 'bytes': size, 'sha256': expected, 'method': method})
    return {'hashed_count': hashed_count, 'files': verified}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('manifest')
    parser.add_argument('quant')
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.manifest, args.quant)))
