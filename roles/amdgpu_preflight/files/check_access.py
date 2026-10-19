from pathlib import Path
import os
import sys

vendor, device = (int(value, 16) for value in sys.argv[1].split(':'))
nodes = []
for render in Path('/sys/class/drm').glob('renderD*'):
    pci = render / 'device'
    if (int((pci / 'vendor').read_text(), 16),
            int((pci / 'device').read_text(), 16)) == (vendor, device):
        nodes.append(Path('/dev/dri') / render.name)
if not nodes:
    raise SystemExit('No render device matches the configured GPU')
for node in [Path('/dev/kfd')] + nodes:
    if not os.access(node, os.R_OK | os.W_OK):
        raise SystemExit(f'Compute user needs read/write access to {node}')
    print(f'Accessible: {node}')
