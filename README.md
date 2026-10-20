# AMD AI Ops

Public AMD GPU recipes for Ubuntu AI developers. RX 6900 XT (`gfx1030`) is the
initial verification card. Manual/Runme and Ansible paths use the same packages
and settings. Generic dependencies and AMD-specific dependencies stay separate.

## Recipes

| Recipe | Manual / Runme | Ansible | Status |
|---|---|---|---|
| ROCm cleanup (optional) | [cleanup.md](cleanup.md) | [cleanup.yml](cleanup.yml) | Automatic discovery and purge verified; rerun made no changes |
| Generic tools, Vulkan, HIP and ROCm PyTorch | [rocm.md](rocm.md) | [site.yml](site.yml) | Installed; basic GPU checks passed, PyTorch requires the documented process-local preload |
| llama.cpp with Vulkan and HIP, Gemma 4 12B | [llama.md](llama.md) | [llama.yml](llama.yml) | Flash Attention verified on both backends; HIP uses the included tile fix |

The base uses Ubuntu packages and the existing distribution `amdgpu` driver.
The separate llama.cpp recipe builds the selected GPU backend and runs one native
model service. Its manual guide separates common setup, Vulkan and HIP.

## Ansible

Run from this repository. `ansible.cfg` selects the ignored `./hosts` inventory:

```bash
cp hosts.example.ini hosts
vim hosts
```

The default SSH user is `ubuntu`; override `ansible_user` in ignored
`host_vars/<inventory_hostname>.yml` when needed. Add `--ask-become-pass` when
sudo requires a password. The optional [cleanup playbook](cleanup.md) discovers
installed ROCm packages. Preview the removals before running it:

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

After verifying the base, review `models` in
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml). This download catalog currently
contains one entry: Gemma 4 12B IT Q4_0, with its name, repository and filename.
Optional `revision` pins a version; optional `sha256` verifies the file. Without
them, downloads use the default revision and existing files are reused.
Set `llama_model` to the model name to serve. Put host-specific settings in ignored
`host_vars/` files.

Build HIP (default) with Flash Attention and start the model:

```bash
ansible-playbook llama.yml --check --diff
ansible-playbook llama.yml --diff
```

Build Vulkan and switch the same service to it, including its inference check:

```bash
ansible-playbook llama.yml --diff -e llama_backend=vulkan
```

Set `llama_backend` in ignored host variables to persist your choice. Without an
override, the next run selects HIP again. Both build directories are retained.
HIP applies the included RDNA2 attention-tile patch before building. The role
removes only that patch before a source update and reapplies it afterward.
Build and service settings live in [roles/llama/defaults/main.yml](roles/llama/defaults/main.yml).
The service listens on
`127.0.0.1:8080`; use an SSH tunnel to access it remotely.

The `model_store` role installs Hugging Face `hf` in `/opt/huggingface` and downloads
the catalog artifacts as the SSH login user. Optional authentication can avoid
anonymous rate limits. Install the CLI first, then run `hf auth login` on the target
as that same user before running the full playbook:

```bash
ansible-playbook llama.yml --diff --tags hf
```

Never store access tokens in the catalog. Existing models in other directories
are not removed.

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
| cleanup 1.3 Verify | rocm_cleanup | Package absence and Vulkan checks |
| cleanup 1.4 Leftovers | rocm_cleanup | `amd.cleanup.files.remove`, `amd.cleanup.directories.remove`, `amd.cleanup.roots.verify` |
| rocm 1.1 Generic tools | development | `development.cache.refresh`, `development.install` |
| rocm 1.2 Vulkan | development | `development.vulkan.install`, `development.vulkan.verify` |
| rocm 2.1 Discovery | rocm | `amd.rocm.cache.refresh`, `amd.rocm.discovery.install`, `amd.rocm.inspect.agents` |
| rocm 2.2 HIP | rocm | `amd.rocm.install`, `amd.rocm.compiler.verify` |
| rocm 2.3 Python | rocm | `amd.rocm.python.install`, `amd.rocm.python.inspect` |
| rocm 2.4 Reboot | rocm | `amd.rocm.reboot.report` (manual reminder) |
| llama 1.1 Configure the model | llama | `llama.settings.require` |
| llama 1.2 Service user | llama | `llama.user.create`, `llama.state.prepare` |
| llama 1.3 Hugging Face CLI | model_store | Install isolated `hf` environment and PATH link |
| llama 1.4 Model | model_store | `llama.model.prepare`, `llama.model.download`, optional checksum and permissions |
| llama 1.5 Source | llama | `llama.source.prepare`, `llama.source.checkout` |
| llama 2.1 Vulkan build | llama | `llama.vulkan.configure`, `llama.vulkan.build`, `llama.backends.verify` |
| llama 3.1 HIP build | llama | `llama.hip.patch`, `llama.hip.configure`, `llama.hip.build`, `llama.backends.verify` |
| llama 2.2 / 3.2 Service | llama | `llama.systemd.write`, `llama.systemd.start` |
| llama 2.3 / 3.3 Inference | llama | `llama.health.verify`, `llama.models.verify`, `llama.inference.verify`, `llama.offload.verify` |

Ansible additionally validates settings and prerequisites. APT installations use
`state: latest`, matching `apt install`, and inherit the host's recommendation
settings. Installation simulations reject removals, kernel changes and non-Ubuntu
install candidates. Cleanup discovers ROCm packages and rejects removals outside
that stack and its required UCX, UCC and Open MPI dependencies.
Check mode uses existing metadata and skips runtime and post-removal checks.
Both installation roles may reuse package metadata up to one hour old.
The llama.cpp role requires the base packages to be installed first. Its check mode
previews source, user and unit changes; source patching, CLI installation, builds, model download and runtime
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

Before the current catalog/role restructuring, llama.cpp `v0.7.0` with the pinned
Gemma 4 12B IT Q4_0 file passed arithmetic and
short Python-code generation through both backend APIs. Logs confirmed 49/49 layers
offloaded to Vulkan0 or ROCm0. The service uses a 4,096-token context and one slot;
this does not validate the model's full native context or multimodal features.

With HIP's default automatic Flash Attention, the first inference aborted at
`GGML_ASSERT(max_blocks_per_sm > 0)` in `fattn-common.cuh`. Disabling Flash Attention
passed the same checks. That baseline used `--flash-attn off` for both backends to keep
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
That baseline's final llama.cpp playbook rerun completed with `changed=0` and
`failed=0`.

On 2026-10-21, the revised roles were deployed on the same RX 6900 XT with
llama.cpp commit `1242be66f` discovered from upstream. These were installation
and API checks, not a rerun of the benchmarks above.

| Step | Result |
|---|---|
| Cleanup without package overrides | Discovered and purged 68 packages; absence checks passed and Vulkan remained available |
| Cleanup rerun | No changes |
| Base reinstall | Passed; `gfx1030`, HIP compiler and PyTorch environment checks passed |
| Vulkan service | Returned `Hello!`; 49/49 layers offloaded to Vulkan0 |
| HIP service | Returned `Hello!`; 49/49 layers offloaded to ROCm0; left active |
| Base and HIP reruns | Both completed with `changed=0`, `failed=0` |

Both backends used Gemma 4 12B IT Q4_0, f16 K/V cache, 4,096 context, one slot,
Flash Attention off and thinking off. The existing model file was reused;
the Hugging Face CLI was installed, but a fresh model download was not exercised.
No reboot or driver replacement was performed.

HIP Flash Attention was then fixed and verified on the same source commit.
The [included patch](roles/llama/files/hip-rdna2-flash-attention.patch) selects
16-column tiles instead of 32 for RDNA2's 256/512-dimensional attention heads.
It preserves `max_blocks_per_sm > 0`; it does not force occupancy to 1.
The suggested assertion override is described as unverified in
[upstream issue #26702](https://github.com/ggml-org/llama.cpp/issues/26702).

| Flash Attention check | Result |
|---|---|
| Unpatched HIP, f16 cache | Reproduced the occupancy assertion |
| Patched HIP, f16 cache | `17 + 25` returned `42` |
| Patched HIP, Q4_0 weights and Q4_0/Q4_0 cache | Correct arithmetic and Python `square()` function |
| Upstream numerical tests against CPU | 4/4 passed: 256/512 heads, decode and prefill |
| Ansible deployment | Patch applied by the role; generation, Flash Attention and 49/49 GPU-layer checks passed |
| Check mode and real rerun | Both passed with `changed=0`, `failed=0` |

Both backends now default to Flash Attention **on**. Both passed the role's
generation and full GPU-offload checks with Flash Attention enabled.
The active service still uses f16 cache, 4,096 context and one slot. Q4_0/Q4_0 was
checked separately with `-ctk q4_0 -ctv q4_0`. These checks do not establish
long-context capacity or a throughput improvement. No GFX override was used.

### Matched Flash Attention benchmark

On 2026-10-21, both backends ran the same Gemma 4 12B IT Q4_0 model with
**Flash Attention on**, comparing f16/f16, Q8_0/Q8_0 and Q4_0/Q4_0 caches.
Both used source `1242be66f`, with the HIP tile fix above.
Settings: pp512/tg128, five repetitions, 10 threads, batch 2048,
microbatch 512 and 99 GPU layers. The service was stopped during each benchmark.

| Backend | Cache K/V | pp512 tokens/s | tg128 tokens/s | Result |
|---|---|---:|---:|---|
| Vulkan | f16/f16 | 1,375.17 ± 1.18 | 52.72 ± 0.14 | Passed |
| HIP | f16/f16 | 1,445.21 ± 15.62 | 51.70 ± 0.06 | Passed |
| Vulkan | Q8_0/Q8_0 | 1,363.12 ± 1.61 | 50.35 ± 0.23 | Passed |
| HIP | Q8_0/Q8_0 | 1,432.80 ± 16.69 | 46.23 ± 0.06 | Passed |
| Vulkan | Q4_0/Q4_0 | 1,340.82 ± 2.02 | 50.21 ± 0.43 | Passed |
| HIP | Q4_0/Q4_0 | 1,432.83 ± 16.50 | 46.97 ± 0.08 | Passed |

The confirmed default is **HIP, f16/f16 cache and Flash Attention on**.
f16/f16 was fastest on both backends in these short-context tests. Vulkan generated
tokens 2.0% faster than HIP; HIP processed prompts 5.1% faster than Vulkan.

The parked `7e15ad0` evaluation compared Vulkan Q8/Q8 against HIP f16/f16;
those rows do not establish a backend speed difference. Its commands and raw
results remain accessible in the external archive. The matched results above
replace that comparison, not the historical measurements.

This stage confirms llama.cpp installation and inference on both backends.
Model optimization is out of scope. These short tests do not assess image quality,
concurrent serving or long-context accuracy.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).
