# Maintainer Notes

## Contributors

Read [AGENTS.md](AGENTS.md) for contribution rules, writing style and command
formatting. Roles define installation and service behavior; inventory supplies
model and host values. This file documents that implementation and maps it to
the manual recipes.

## Layout and Ownership

| File | Purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Contribution rules and temporary internal evidence pointers |
| [README.md](README.md) | Quick starts, compatibility, acceptance criteria and a short changelog |
| [MODELS.md](MODELS.md) | Model descriptions, file sizes and complete Ansible/Runme profiles for every model, including the current default |
| [BENCHMARK.md](BENCHMARK.md) | Measured settings, results, comparisons, conclusions and limitations |
| [reports/](reports/) | `index.html` is the entry point and owns recommendations; `report.html` shows model tests and `mtp.html` shows MTP/users. All three share navigation and `style.css`; no independent configuration guidance |
| [evaluation/](evaluation/) / [evaluation.yml](evaluation.yml) | Evaluation runner, scoring inputs and report-generation sources; HTML output lives in `reports/`, raw logs stay outside the repository |
| [cleanup.md](cleanup.md) / [cleanup.yml](cleanup.yml) | Optional ROCm cleanup |
| [rocm.md](rocm.md) / [site.yml](site.yml) | Generic development tools, then AMD libraries |
| [llama.md](llama.md) / [llama.yml](llama.yml) | Current default model recipe and role entry point; alternatives link to MODELS.md |
| [roles/](roles/) | Source of truth for task order, supported settings, service arguments and verification |
| [ansible.cfg](ansible.cfg) / [hosts.example.ini](hosts.example.ini) | Select ignored `./hosts` inventory / provide its template |
| [group_vars/all.yml](group_vars/all.yml) | Default SSH user |
| [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml) | Active Ansible values: hardware, packages, model catalog and serving selection; documentation mirrors these values |
| `host_vars/` | Ignored settings for individual hosts |

Keep each model's complete configuration in MODELS.md when the default changes.
The default profile is intentionally repeated in llama.md so the recipe works
on its own. Keep both copies aligned with role behavior and inventory values.

## Role Behavior

- `development` installs generic build/Python tools and Vulkan dependencies.
- `rocm` validates the distribution driver and GPU access, installs Ubuntu HIP
  and PyTorch packages, and checks the compiler and Python environment. The
  PyTorch preload applies only to its Python command, not the system loader.
- `rocm_cleanup` discovers installed ROCm packages and leftover package
  configuration. It permits required UCX, UCC and Open MPI removals, rejects
  unrelated removals and driver changes, and does not run autoremove. Optional
  file cleanup removes only explicitly selected, unowned files and empty directories.
- `model_store` installs `hf` and downloads inventory's `models` list as the SSH
  login user. Authentication belongs to that user; files remain readable by the
  service. Existing files are reused unless an optional checksum requires replacement.
  Downloads use the optional `revision`, or the repository default.
- `llama` creates the service user before invoking `model_store`, then builds the
  selected backend and serves the entry named by `llama_model`. Backend builds
  use separate directories; only one localhost systemd service runs at a time.
  Model artifacts and selection belong in inventory, not role defaults.
- `llama_evaluation` runs the requested matrix serially with supplied model paths.
  It preserves completed results and attempts to restart the service if it was
  active before the run. Workflow and scoring: [evaluation/README.md](evaluation/README.md).
  HIP FA occupancy failures get separate FA-off retries. Quantized V-cache
  combinations rejected without FA are recorded as unsupported.

Optional `models[].mmproj` uses the model's repository and revision. Only BF16
projector filenames are accepted; this does not inspect tensor precision.
`mmproj_sha256` optionally verifies the artifact. Optional image minimum/maximum
token settings must be supplied together and fit within `llama_ubatch`;
`llama_batch` must be at least `llama_ubatch`. Model-specific image budgets
live in inventory and [MODELS.md](MODELS.md). When a projector is enabled,
the role checks that it loads on the selected GPU backend.

Source updates follow the upstream default branch unless overridden. The HIP
Flash Attention patch applies to `gfx103x`; its kernel also checks the GPU family.
It uses 4-column tiles for 72-dimensional projector heads and smaller tiles for
256/512-dimensional heads, preserving the occupancy assertion. Before a source
or patch update, remove only the installed managed patch, then
reapply it when required. Stop if the patch no longer applies.

## Step Map

Each recipe's `ops-step` names its Ansible task. Grouped cells map to several
tasks; `.hip` IDs distinguish cells using the same backend-neutral task.
HIP section 3.3 reuses the verification cell in section 2.3.

Both paths follow setup, build, service and verification. The role handles
patching in `source.yml`; the manual shows it beside the HIP build. Manual
service cells restart immediately; Ansible restarts only on changes.
Validation and check-mode helpers remain Ansible-only.

Update roles, recipe markers and this map together.

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
| llama 1.1 Configure the model | [llama/main.yml](roles/llama/tasks/main.yml) | `llama.settings.require`; values from role defaults and inventory |
| llama 1.2 Service user | [llama/main.yml](roles/llama/tasks/main.yml) | `llama.user.create`, `llama.state.prepare` |
| llama 1.3 Hugging Face CLI | [model_store/main.yml](roles/model_store/tasks/main.yml) | `model_store.hf.install`, `model_store.hf.link` |
| llama 1.4 Model and optional BF16 projector | [model_store/download.yml](roles/model_store/tasks/download.yml), [artifact.yml](roles/model_store/tasks/artifact.yml) | `llama.model.prepare`, `llama.model.download`, `model_store.artifact.permissions`, optional `model_store.artifact.verify` |
| llama 1.5 Source | [llama/source.yml](roles/llama/tasks/source.yml) | `llama.source.prepare`, `llama.source.checkout`; managed patch removal before updates |
| llama 2.1 Vulkan build | [llama/main.yml](roles/llama/tasks/main.yml) | `llama.vulkan.configure`, `llama.vulkan.build`, `llama.backends.verify` |
| llama 3.1 HIP build | [llama/source.yml](roles/llama/tasks/source.yml), [main.yml](roles/llama/tasks/main.yml) | `llama.hip.patch`, `llama.compiler.locate`, `llama.hip.configure`, `llama.hip.build`, `llama.backends.verify` |
| llama 2.2 / 3.2 Service | [llama/main.yml](roles/llama/tasks/main.yml), [service template](roles/llama/templates/llama.service.j2) | `llama.systemd.write`, `llama.systemd.reload`, `llama.systemd.start` |
| llama 2.3 / 3.3 Shared verification | [llama/main.yml](roles/llama/tasks/main.yml) | `llama.health.verify`, `llama.models.verify`, `llama.inference.verify`, `llama.service.inspect`, `llama.offload.verify` (including FA and optional projector placement) |
| llama 4.1 Image example | Manual only | Describe a supplied image and review the answer; the role checks projector loading, not image-answer quality |

## Validation and Check Mode

Installation roles validate prerequisites and settings, use `state: latest` and
inherit APT's recommendation settings. Package simulations reject removals,
kernel changes and non-Ubuntu candidates. Cached package metadata may be reused
for up to one hour. Kernel drivers remain unchanged; reboot is manual.

Check mode uses existing metadata and skips runtime and post-removal checks.
For `llama`, it previews source, user and service changes. Patching, CLI
installation, compilation, downloads and inference require a real run and an
installed base stack.

A real deployment must pass workload checks and an unchanged rerun before it is
reported as validated. See [README acceptance criteria](README.md#compatibility-and-acceptance)
and record results in [BENCHMARK.md](BENCHMARK.md).
