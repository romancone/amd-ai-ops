# AMD AI Ops

Public AMD GPU recipes for Ubuntu AI developers. RX 6900 XT (`gfx1030`) is the
initial verification card. Manual/Runme and Ansible paths use the same packages
and settings. Generic dependencies and AMD-specific dependencies stay separate.

## Recipes

| Recipe | Manual / Runme | Ansible | Status |
|---|---|---|---|
| Reviewed legacy ROCm cleanup (optional) | [cleanup.md](cleanup.md) | [cleanup.yml](cleanup.yml) | Verified purge, leftover removal and idempotence on the verification host |
| Generic tools, Vulkan, HIP and ROCm PyTorch | [rocm.md](rocm.md) | [site.yml](site.yml) | Installed; basic GPU checks passed, PyTorch requires the documented process-local preload |

The base uses Ubuntu packages and the existing distribution `amdgpu` driver.
Application recipes remain deferred; this repository change covers the base installation and cleanup.

## Ansible

Run from this repository. `ansible.cfg` selects the ignored `./hosts` inventory:

```bash
cp hosts.example.ini hosts
vim hosts
```

The default SSH user is `ubuntu`; override `ansible_user` in ignored
`host_vars/<inventory_hostname>.yml` when needed. Add `--ask-become-pass` when
sudo requires a password. Configure an explicit cleanup profile as described in
[cleanup.md](cleanup.md) before using the optional cleanup playbook:

```bash
ansible-playbook cleanup.yml --check --diff
ansible-playbook cleanup.yml --diff
```

Preview and install the base:

```bash
ansible-playbook site.yml --check --diff
ansible-playbook site.yml --diff
```

Reboot the host manually, reconnect and repeat the GPU checks. Settings live in
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml) and role defaults. Keep manual
package lists and settings in sync when overriding them.

Local checks do not connect to hosts:

```bash
ansible-playbook cleanup.yml --syntax-check --diff
ansible-playbook cleanup.yml --list-hosts --diff
ansible-playbook site.yml --syntax-check --diff
ansible-playbook site.yml --list-hosts --diff
```

## Manual / Runme

Open the notebooks in a VS Code session on the target host. Run named Bash cells
in order, stopping on any failed command. Review APT's proposal; resolve removals,
kernel changes and third-party ROCm conflicts separately. The same commands work
in a terminal.

```bash
runme list --filename cleanup.md
runme list --filename rocm.md
```

## Step Map

| Notebook part | Role | Operations |
|---|---|---|
| cleanup 1.2 Purge | rocm_cleanup | `amd.cleanup.packages.remove` |
| cleanup 1.3 Verify | rocm_cleanup | `amd.cleanup.roots.verify`, package absence and Vulkan checks |
| rocm 1.1 Generic tools | development | `development.cache.refresh`, `development.install` |
| rocm 1.2 Vulkan | development | `development.vulkan.install`, `development.vulkan.verify` |
| rocm 2.1 Discovery | rocm | `amd.rocm.cache.refresh`, `amd.rocm.discovery.install`, `amd.rocm.inspect.agents` |
| rocm 2.2 HIP | rocm | `amd.rocm.install`, `amd.rocm.compiler.verify` |
| rocm 2.3 Python | rocm | `amd.rocm.python.install`, `amd.rocm.python.inspect` |
| rocm 2.4 Reboot | rocm | `amd.rocm.reboot.report` (manual reminder) |

Ansible additionally validates settings and prerequisites. APT installations use
`state: latest`, matching `apt install`, and inherit the host's recommendation
settings. Installation simulations reject removals, kernel changes and non-Ubuntu
install candidates. Cleanup rejects removals beyond its explicit package selection.
Check mode uses existing metadata and skips runtime and post-removal checks.
Both installation roles may reuse package metadata up to one hour old.

## Compatibility and Acceptance

Sources reviewed on 2026-10-20:
[Ubuntu ROCm guidance](https://ubuntu.com/blog/amd-rocm-on-ubuntu),
[ROCm development packages](https://packages.ubuntu.com/resolute/rocm-dev),
[ROCm PyTorch](https://packages.ubuntu.com/resolute/python3-torch-rocm),
[AMD product matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html),
[Vulkan build dependencies](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan).

Ubuntu 26.04 supplies ROCm 7.1 and ROCm-enabled PyTorch 2.9.1. AMD's current product
matrix does not explicitly list RX 6900 XT. Package availability and the shared
`gfx1030` architecture do not establish vendor product or framework support.

On 2026-10-20, the RX 6900 XT passed a compiled Vulkan compute shader and a compiled
HIP kernel, each checking 1,024 results. PyTorch `2.9.1+debian` (Ubuntu package
`2.9.1+dfsg-1~exp2ubuntu2`) passed GPU matrix multiplication against a CPU reference,
forward/backward computation, finite-gradient checks and an SGD optimizer update
with the process-local preload below. No GPU driver replacement or reboot was used
for these checks. Application workloads remain unvalidated.
The final base playbook rerun completed with `changed=0` and `failed=0`.

Without the preload, GPU allocation succeeded but tensor transfer failed with
`HIP error: invalid argument`; a zero-fill diagnostic also crashed. HIP diagnostics
reported `Unable to find target for this triple (no targets are registered)` while
creating internal blit kernels. Exported LLVM target symbols in `libtorch_cpu` and
the successful preload trial point to a library-loading conflict. This is a local
diagnosis, not an upstream-confirmed fix.

Prefix PyTorch commands with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libamd_comgr.so.3`.
The role scopes the same setting to its Python command; it does not alter shell
profiles or the system loader. Retest without this workaround after package updates.
The basic checks do not establish compatibility with every training workload.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).
