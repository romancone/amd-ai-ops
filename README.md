# AMD AI Ops

Public AMD GPU recipes for Ubuntu AI developers. RX 6900 XT (`gfx1030`) is the
initial verification card. Manual/Runme and Ansible paths use the same packages
and settings. Generic dependencies and AMD-specific dependencies stay separate.

## Recipes

| Recipe | Manual / Runme | Ansible |
|---|---|---|
| ROCm cleanup (optional) | [cleanup.md](cleanup.md) | [cleanup.yml](cleanup.yml) |
| Generic tools, Vulkan, HIP and ROCm PyTorch | [rocm.md](rocm.md) | [site.yml](site.yml) |
| llama.cpp with Vulkan and HIP, Gemma 4 12B | [llama.md](llama.md) | [llama.yml](llama.yml) |

The base uses Ubuntu packages and the existing distribution `amdgpu` driver.
The llama.cpp recipe builds the selected GPU backend and runs one model service.

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
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml).

After verifying the base, choose a [model configuration](MODELS.md) and apply
it in [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml).

Optional: save your backend choice in `host_vars/<inventory_hostname>.yml`:

```yaml
llama_backend: hip  # hip (default) or vulkan
```

Run one command to build the backend, start the service with Flash Attention and
check inference:

```bash
# HIP
ansible-playbook llama.yml --diff -e llama_backend=hip

# Vulkan
ansible-playbook llama.yml --diff -e llama_backend=vulkan
```

Without `-e llama_backend=...`, Ansible uses your saved choice, or HIP by default:

```bash
ansible-playbook llama.yml --diff
```

Build and service settings live in [roles/llama/defaults/main.yml](roles/llama/defaults/main.yml).
The service listens on
`127.0.0.1:8080`; use an SSH tunnel to access it remotely.

Optional: sign in to Hugging Face to reduce download throttling. Install `hf`
first, then run `hf auth login` on the target as your SSH user before deployment:

```bash
ansible-playbook llama.yml --diff --tags hf
```

Keep access tokens out of inventory files.

## Manual / Runme

Runme is an optional VS Code extension that lets you click Run on a guide's Bash
blocks instead of copying and pasting commands. You can also copy the same
commands into a terminal.

Install the **Runme** extension, connect VS Code to the target host through
**Remote SSH**, and open the guide as a Runme notebook. Run its cells in order
as your normal user, checking each result before continuing. Stop if a command
fails. Review APT's proposal; resolve removals, kernel changes and third-party
ROCm conflicts separately.

```bash
runme list --filename cleanup.md
runme list --filename rocm.md
runme list --filename llama.md
```

Recipe-to-role mapping and Ansible behavior: [Maintainer Notes](MAINTAINERS.md#step-map).

## Compatibility and Acceptance

Verified platform: Ubuntu 26.04 with RX 6900 XT (`gfx1030`). Local verification
is not an AMD support guarantee; consult the
[AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
for vendor support.

PyTorch requires the per-command `LD_PRELOAD` setting shown in
[rocm.md](rocm.md#23-amd-pytorch-for-python). Ansible applies it to its Python check.
Do not apply it to llama.cpp.

Acceptance criteria:

| Component | Required result |
|---|---|
| Cleanup | Discovered ROCm packages are removed; the GPU driver and Vulkan still work. |
| Vulkan and HIP | Compiled GPU workloads return the expected results on both backends. |
| PyTorch | GPU matrix multiplication matches the CPU reference; gradients are finite and an optimizer update succeeds. |
| llama.cpp | Both backends serve the selected model, generate a correct test answer and offload all model layers with Flash Attention on. |
| Images (optional) | The BF16 projector loads on the selected GPU backend and the response identifies the image's main content. |
| Ansible rerun | An unchanged setup finishes without failures or changes. |

These basic checks do not establish long-context accuracy, concurrent serving
capacity or compatibility with every training workload.

## Models and Benchmarks

- [Models](MODELS.md): descriptions, download sizes and configuration examples.
- [Benchmarks](BENCHMARK.md): measurements and conclusions.
- [HTML summary](reports/index.html) and [interactive results](reports/report.html): comparisons and saved image answers.
- [Evaluation workflow](evaluation/README.md): run benchmarks, score answers and generate reports.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).

## Changelog

### 2026-10-25 — Q8 model evaluation

- Add optional full-context checks and GPU memory limits to evaluation.
- Publish 72 configurations, Q4/Q8 image-quality comparisons and 8/16 GiB recommendations.
- Simplify HTML reports and share duplicate answers without removing results.

### 2026-10-24 — [79dc65d](https://github.com/romancone/amd-ai-ops/commit/79dc65d)

- Add E4B and Qwen model profiles, 16K configuration and aligned Runme settings.
- Fix RDNA2 HIP Flash Attention for Qwen image projectors.
- Add repeatable evaluation, Luna-6 quality scoring and HTML comparisons.

Service checks on **2026-10-23** — 16K HIP/Vulkan passed;
model measurements are in [BENCHMARK.md](BENCHMARK.md).
