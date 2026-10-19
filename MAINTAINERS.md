# Maintainer Notes

This repository contains public AMD GPU recipes. The RX 6900 XT is the initial
verification card; select hardware by its ROCm architecture target (`gfx1030`).
Keep paired operator paths with the same settings, installation order, and result:

- Concise Markdown commands for manual use and Runme.
- Reusable Ansible roles with prerequisite, settings, package and runtime validation.

## Layout and Ownership

- `README.md` owns quick starts, recipe mapping, and delivery status.
- `AGENTS.md` owns contribution and recipe conventions.
- `rocm.md` is the manual pair for `site.yml` and the `rocm` role.
- `ansible.cfg` selects the ignored `./hosts` inventory.
- `hosts.example.ini` is the public inventory template; local `host_vars/` stay ignored.
- `group_vars/all.yml` sets SSH user `ubuntu`.
- `group_vars/amd_hosts.yml` holds the hardware and package profile for `amd_hosts`.

## Current Recipe

The `rocm` role validates the existing distribution driver and compute access,
installs `rocminfo`, verifies GPU enumeration, and installs Ubuntu's `rocm-dev`.
It verifies compiler availability and leaves kernel drivers unchanged.

Status: prepared; not deployed or workload-validated. Application deployment is
separate from this library installation recipe.

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
