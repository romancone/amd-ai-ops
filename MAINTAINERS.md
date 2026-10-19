# Maintainer Notes

This repository is the public home for reusable AMD AI host operations, starting
with the Radeon RX 6900 XT.

Keep two operator paths with the same installation result:

- Ansible roles for repeatable deployment.
- Runme-friendly Markdown recipes for human review and manual execution.

Manual recipes keep essential safeguards and final verification. Ansible roles
also include complete prerequisite/settings validation, runtime checks,
idempotence, and check-mode handling. Manual recipes need not reproduce
validation-only Ansible tasks or automation bookkeeping.

## Layout and Ownership

- `README.md` explains the project, quick starts, recipe mapping, and delivery status.
- `AGENTS.md` owns contribution, numbering, alignment, and validation rules.
- `ansible.cfg` selects the ignored `./hosts` inventory; run from the repository root.
- `hosts.example.ini` is the public inventory template. Real inventory and local
  `host_vars/` files stay ignored, including non-default SSH users and cleanup lists.
- `group_vars/all.yml` sets SSH user `ubuntu`; `group_vars/amd_hosts.yml` defines
  the shared hardware/package profile.
- `site.yml` targets `amd_hosts` and composes reusable roles from `roles/`.
- `rocm.md` contains the manual procedure, settings, safeguards, and final checks.

## Current Recipe

`rocm.md` is paired with `site.yml`, in this role order:

1. `amdgpu_preflight`: require the distribution driver and compute-user access.
2. `rocm`: install discovery, review explicit legacy cleanup, and install `rocm-dev`.
3. `rocm_smoke_test`: compile and execute a temporary HIP test.

Status: prepared, not deployed or GPU-validated. All three roles are included by
default; the discovery tag stops before cleanup and development-profile installation.
An inference application recipe remains a separate deliverable.

## Recipe Alignment

Number manual sections by playbook role order and group each logical part into
one shell block. Keep stable `ops-step` IDs and `ansible_task` mappings for every
included operation; several task mappings can precede one grouped command block.
Validation-only tasks stay in Ansible. Update both paths when shared settings,
installation order, safeguards, or verification change.

Keep roles generic. Hardware, package sources, versions, and migration choices
belong in variables. Keep kernel driver and firmware decisions separate from
ROCm user-space and application dependencies.

## Migration and Validation

The legacy cleanup list is empty by default. Configured removals require matching
installed versions and a reviewed APT simulation. Reject unrelated removals,
installed-package upgrades, kernel/driver installs, and non-Ubuntu origins.
Never purge, autoremove, or manually delete `/opt`. Cleanup and profile installation
are separate transactions; recovery details belong in `rocm.md`.

Check mode reports deferred verification when discovery tools are missing and
skips HIP execution. Ansible always cleans the temporary smoke-test workspace;
these transient actions report unchanged. Package-cache refresh can report a
change after its 3600-second validity window.

Offline tests check included-operation mappings, recipe numbering, shell syntax,
shared test sources, and transaction guards. They do not establish GPU support.
Require target enumeration, HIP execution, an intended application workload, and
an idempotence run before claiming a validated configuration. Record software
versions, commands, results, and evidence dates; keep volatile Git state out of docs.
