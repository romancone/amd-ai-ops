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
- `ansible.cfg` selects the ignored `./hosts` inventory.
- `hosts.example.ini` is the public inventory template; local `host_vars/` stay ignored.
- `group_vars/all.yml` sets SSH user `ubuntu`.
- `group_vars/amd_hosts.yml` holds the hardware and package profile for `amd_hosts`.

## Current Recipe

The `development` role installs generic build/Python tools and Vulkan dependencies.
The `rocm` role validates the existing distribution driver and compute access,
installs `rocminfo`, verifies GPU enumeration, and installs Ubuntu's `rocm-dev`,
monitoring tools and ROCm PyTorch. It verifies compiler availability and reports
the Python environment. Kernel drivers stay unchanged.

Legacy cleanup is opt-in with an exact package/version map in inventory variables.
It rejects extra APT removals, avoids autoremove and only removes explicitly selected
unowned regular files and empty directories. It stops on other leftovers.
No host-specific cleanup selection belongs in role defaults.

Base installation and Vulkan/HIP compute are verified on the initial card.
Basic PyTorch training is verified only with the process-local compiler preload
documented in README. No global loader change is made. Application recipes remain deferred.

## Recipe Alignment

Use two-level numbering in Markdown, with one top-level section per playbook role.
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
