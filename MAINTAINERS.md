# Maintainer Notes

## Contributors

Read [AGENTS.md](AGENTS.md) for contribution rules, writing style and command
formatting. Use this file as the source of truth for repository layout,
recipe-to-role mapping and essential Ansible behavior. Update it when those
details change; link here instead of repeating them in operator recipes.

## Layout and Ownership

| File | Purpose |
|---|---|
| [README.md](README.md) | Quick starts, compatibility and acceptance criteria |
| [MODELS.md](MODELS.md) | Model state, recommendations and YAML examples; deployment steps belong in `llama.md` |
| [BENCHMARK.md](BENCHMARK.md) | Measurements, test results and limitations |
| [cleanup.md](cleanup.md) / [cleanup.yml](cleanup.yml) | Optional ROCm cleanup |
| [rocm.md](rocm.md) / [site.yml](site.yml) | Generic development tools, then AMD libraries |
| [llama.md](llama.md) / [llama.yml](llama.yml) | Model service after the base install; common setup, Vulkan and HIP |
| [ansible.cfg](ansible.cfg) / [hosts.example.ini](hosts.example.ini) | Select ignored `./hosts` inventory / provide its template |
| [group_vars/all.yml](group_vars/all.yml) | Default SSH user |
| [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml) | Hardware, packages, `models` catalog and `llama_model` selection |
| `host_vars/` | Ignored settings for individual hosts |

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

Optional `models[].mmproj` uses the model's repository and revision. Only BF16
projector filenames are accepted; this does not inspect tensor precision.
`mmproj_sha256` optionally verifies the artifact. When enabled, the role checks
that the projector loads on the selected GPU backend.

Source updates follow the upstream default branch unless overridden. The HIP
Flash Attention patch applies to `gfx103x`; its kernel also checks the GPU family.
It uses smaller attention tiles for 256/512-dimensional heads and preserves the
occupancy assertion. Before an update, remove only the managed patch, then
reapply it when required. Stop if the patch no longer applies.

## Step Map

Update this map with any change to recipe sections, operation IDs or role
ownership. Keep recipes and tasks aligned in the same change. Validation-only
tasks remain in Ansible.

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
| llama 1.4 Model and optional BF16 projector | model_store | `llama.model.prepare`, `llama.model.download`, optional checksums and permissions |
| llama 1.5 Source | llama | `llama.source.prepare`, `llama.source.checkout` |
| llama 2.1 Vulkan build | llama | `llama.vulkan.configure`, `llama.vulkan.build`, `llama.backends.verify` |
| llama 3.1 HIP build | llama | `llama.hip.patch`, `llama.hip.configure`, `llama.hip.build`, `llama.backends.verify` |
| llama 2.2 / 3.2 Service | llama | `llama.systemd.write`, `llama.systemd.start` |
| llama 2.3 / 3.3 Inference | llama | `llama.health.verify`, `llama.models.verify`, `llama.inference.verify`, `llama.offload.verify` |
| llama 4.1 Image example | Manual | Optional image request; visual review by the operator |

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
