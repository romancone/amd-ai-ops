# Maintainer Notes

This repository contains public AMD GPU recipes. The RX 6900 XT is the initial
verification card; select hardware by its ROCm architecture target (`gfx1030`).
Keep paired operator paths with the same settings, installation order, and result:

- Concise Markdown commands for manual use and Runme.
- Reusable Ansible roles with prerequisite, settings, package and runtime validation.

## Layout and Ownership

- `README.md` owns quick starts, recipe mapping, and delivery status.
- `AGENTS.md` owns contribution and recipe conventions.
- `rocm.md` pairs with `site.yml`: generic `development` first, then AMD-specific `rocm`.
- `cleanup.md` pairs with the separate `cleanup.yml` and reusable `rocm_cleanup` role.
- `llama.md` pairs with `llama.yml` after the base install: common setup, Vulkan, HIP.
- Inventory's `models` list owns model names, repositories and filenames; revisions and checksums are optional.
- `model_store` installs `hf` and downloads the list; `llama` serves the entry named by `llama_model`.
- Optional `models[].mmproj` selects a publisher-provided BF16 projector from the same repository and revision. Other projector precisions are rejected by filename; optional `mmproj_sha256` verifies the chosen artifact.
- `ansible.cfg` selects the ignored `./hosts` inventory.
- `hosts.example.ini` is the public inventory template; local `host_vars/` stay ignored.
- `group_vars/all.yml` sets SSH user `ubuntu`.
- `group_vars/amd_hosts.yml` holds hardware, packages, the model catalog and service model selection for `amd_hosts`.

## Current Recipe

The `development` role installs generic build/Python tools and Vulkan dependencies.
The `rocm` role validates the existing distribution driver and compute access,
installs `rocminfo`, verifies GPU enumeration, and installs Ubuntu's `rocm-dev`,
monitoring tools and ROCm PyTorch. It verifies compiler availability and reports
the Python environment. Kernel drivers stay unchanged.

Cleanup discovers ROCm packages from installed source-package metadata, including
leftover package configuration. It permits APT-required UCX, UCC and Open MPI
removals but rejects other dependencies and driver changes. It avoids autoremove.
Optional leftover files and directories remain explicit inventory selections;
only unowned regular files and empty directories are removed.

Base installation and Vulkan/HIP compute are verified on the initial card.
Basic PyTorch training is verified only with the process-local compiler preload
documented in README. No global loader change is made.

The llama.cpp role follows the upstream default branch and builds the selected backend in its own
directory. HIP is the default; one localhost systemd service runs at a time. Model details belong in inventory's
`models` list, not role defaults. `llama_model` selects one entry by name; it is a
string, not the model configuration object. The catalog currently contains one
Gemma 4 12B Q4_0 entry, and the manual uses that same example throughout.
Without `sha256`, existing model files are reused. A new download uses the supplied
`revision`, or the repository default when omitted.
Create the service user before invoking `model_store`. Downloads belong to the
login user and remain readable by the service. `hf` authentication also belongs
to the login user; never put tokens in the catalog.
The selected backend must pass API generation and full GPU-layer offload checks.
The confirmed baseline is HIP with f16/f16 cache and Flash Attention on.
Vulkan also enables Flash Attention. The HIP patch applies only to RDNA2
targets matching `gfx103x`; the patched kernel also checks the runtime GPU family.
The patch limits 256/512-dimensional heads to smaller tiles and preserves the
occupancy assertion. Remove only this managed patch before updating the source,
then reapply it for HIP on RDNA2. Stop if an upstream change makes it incompatible.
Keep the manual commands and service template aligned.
Optional BF16 projector loading and a basic image request passed on HIP and Vulkan.
The role checks projector GPU placement when enabled. The default microbatch caps
Gemma's image budget at 512 tokens; broader visual quality remains unverified.
Do not apply the Python compiler preload to this standalone C++ service.

## Recipe Alignment

Use two-level numbering in Markdown. The llama guide groups shared setup first,
then gives separate Vulkan and HIP paths; other recipes follow playbook role order.
Keep stable `ops-step` IDs for included operations. Validation-only tasks remain
in Ansible; manual recipes do not reproduce its internal checks or bookkeeping.

The manual recipe uses named Bash cells for Runme, standard Ubuntu commands,
and one block per numbered logical part. Keep controller
commands in README and target-host commands in the recipe. README's step map
connects notebook cells to role operations; keep it current when grouping changes.

Use native Ansible modules and direct tool invocations. Keep roles host-agnostic.
Do not add tests, embedded scripts, or host-specific package cleanup.

## Validation

Use local playbook syntax and inventory checks. Both paths install or update the
requested packages using the host's APT recommendation settings. The role rejects
package removals, driver/kernel changes, and non-Ubuntu package installs.
It skips GPU enumeration and compiler execution in check mode. A real application
workload and an idempotence run are required before claiming a validated setup.
