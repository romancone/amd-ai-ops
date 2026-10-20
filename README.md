# AMD AI Ops

Public AMD GPU recipes for Ubuntu AI developers. RX 6900 XT (`gfx1030`) is the
initial verification card. Manual/Runme and Ansible paths use the same packages
and settings. Generic dependencies and AMD-specific dependencies stay separate.

## Recipes

| Recipe | Manual / Runme | Ansible | Status |
|---|---|---|---|
| Reviewed legacy ROCm cleanup (optional) | [cleanup.md](cleanup.md) | [cleanup.yml](cleanup.yml) | Verified purge, leftover removal and idempotence on the verification host |
| Generic tools, Vulkan, HIP and ROCm PyTorch | [rocm.md](rocm.md) | [site.yml](site.yml) | Installed; basic GPU checks passed, PyTorch requires the documented process-local preload |
| llama.cpp with Vulkan and HIP, Gemma 4 12B | [llama.md](llama.md) | [llama.yml](llama.yml) | Text generation verified on both; Flash Attention disabled |

The base uses Ubuntu packages and the existing distribution `amdgpu` driver.
The separate llama.cpp recipe builds both GPU backends and runs one native model
service, following the NVIDIA vLLM recipe's installation and API-verification layout.

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

After verifying the base, build both llama.cpp backends and start Gemma 4 12B Q4_0:

```bash
ansible-playbook llama.yml --check --diff
ansible-playbook llama.yml --diff
```

Vulkan is the default. Switch the same service to HIP and repeat its inference check:

```bash
ansible-playbook llama.yml --diff -e llama_backend=hip
```

Set `llama_backend` in ignored host variables to persist your choice. Without an
override, the next run selects Vulkan again. Settings and pinned revisions live in
[roles/llama/defaults/main.yml](roles/llama/defaults/main.yml). The service listens on
`127.0.0.1:8080`; use an SSH tunnel to access it remotely.

Local checks do not connect to hosts:

```bash
ansible-playbook cleanup.yml --syntax-check --diff
ansible-playbook cleanup.yml --list-hosts --diff
ansible-playbook site.yml --syntax-check --diff
ansible-playbook site.yml --list-hosts --diff
ansible-playbook llama.yml --syntax-check --diff
ansible-playbook llama.yml --list-hosts --diff
```

## Manual / Runme

Open the notebooks in a VS Code session on the target host. Run named Bash cells
in order, stopping on any failed command. Review APT's proposal; resolve removals,
kernel changes and third-party ROCm conflicts separately. The same commands work
in a terminal.

```bash
runme list --filename cleanup.md
runme list --filename rocm.md
runme list --filename llama.md
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
| llama 1.1 Settings | llama | `llama.settings.require` |
| llama 1.2 Source | llama | `llama.source.prepare`, `llama.source.checkout` |
| llama 1.3 Vulkan build | llama | `llama.vulkan.configure`, `llama.vulkan.build` |
| llama 1.4 HIP build | llama | `llama.hip.configure`, `llama.hip.build`, `llama.backends.verify` |
| llama 1.5 Model | llama | `llama.model.prepare`, `llama.model.download` |
| llama 1.6 Service user | llama | `llama.user.create`, `llama.state.prepare` |
| llama 1.7 Service | llama | `llama.systemd.write`, `llama.systemd.start` |
| llama 1.8 Inference | llama | `llama.health.verify`, `llama.models.verify`, `llama.inference.verify`, `llama.offload.verify` |

Ansible additionally validates settings and prerequisites. APT installations use
`state: latest`, matching `apt install`, and inherit the host's recommendation
settings. Installation simulations reject removals, kernel changes and non-Ubuntu
install candidates. Cleanup rejects removals beyond its explicit package selection.
Check mode uses existing metadata and skips runtime and post-removal checks.
Both installation roles may reuse package metadata up to one hour old.
The llama.cpp role requires the base packages to be installed first. Its check mode
previews source, user and unit changes; builds, the model download and runtime
verification require a real run.

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
for these checks.
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

llama.cpp `v0.7.0` with the pinned Gemma 4 12B IT Q4_0 file passed arithmetic and
short Python-code generation through both backend APIs. Logs confirmed 49/49 layers
offloaded to Vulkan0 or ROCm0. The service uses a 4,096-token context and one slot;
this does not validate the model's full native context or multimodal features.

With HIP's default automatic Flash Attention, the first inference aborted at
`GGML_ASSERT(max_blocks_per_sm > 0)` in `fattn-common.cuh`. Disabling Flash Attention
passed the same checks. The recipe sets `--flash-attn off` for both backends to keep
a common tested baseline; this is specific to the tested release, model and card.
The C++ service does not use the PyTorch compiler preload.

Default-size `llama-bench` verification used pp512 and tg128, five repetitions,
f16 KV cache, all layers on the selected GPU and Flash Attention off:

| Backend | pp512 tokens/s | tg128 tokens/s |
|---|---:|---:|
| Vulkan | 1,327.14 ± 1.49 | 52.14 ± 0.20 |
| HIP | 1,221.53 ± 8.10 | 48.71 ± 0.03 |

Both exited successfully and reached 99% GPU utilization and 281 W in samples
collected every 0.5 seconds. Earlier, longer pp512/tg256 tests with ten repetitions
ran for 57 and 60 seconds respectively, with mean GPU utilization of 91.3% and
95.1%. These are short-context benchmarks, not serving capacity at a full
4,096-token context. Subsecond API checks are easy to miss in `nvtop`.

An additional HIP trial with the existing Google Gemma 4 12B IT QAT Q4_0 model,
Q4_0 K/V caches and Flash Attention on also aborted at the same occupancy assertion.
The old `GGML_HIP_ROCWMMA_FATTN` and `GGML_FLASH_ATTN` CMake variables are not
consumed by the pinned source; current HIP builds already include attention kernels.
Do not add these flags or claim that the previous 131,072-token, two-slot multimodal
configuration has been validated by these tests.
The final llama.cpp playbook rerun completed with `changed=0` and `failed=0`.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).
