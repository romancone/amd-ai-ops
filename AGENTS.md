# Project Rules

## Git

- Use `codex/<topic>` feature branches for new work. If a branch change has not
  already been requested, propose its name for human confirmation before creating it.
- Never commit or push without explicit human authorization.
- Keep commits reviewable and scoped to one feature or cleanup.

## Writing Style

- Do not bloat comments or write reasoning and history into code comments.
- Do not use ASCII decoration.
- Public docs are for humans first: concise, practical, and current.

## Recipes

- Keep Ansible roles reusable and host-agnostic.
- Align manual recipes with their Ansible roles in settings, installation order,
  and final result.
- Keep manual recipes simple. Skip redundant variable, OS, and preflight checks,
  change tracking, and check-mode reports. Retain essential safeguards and final
  verification.
- Ansible roles must include complete prerequisite and settings validation,
  runtime checks, idempotence, and check-mode handling.
- Use two-level numbering in Markdown recipes: `## 1. Role name`, then
  `### 1.1 Logical part`, `### 1.2 Logical part`, and so on.
- Match top-level numbers to the roles in playbook order. For manual-only
  recipes, use one top-level service section and number its logical parts.
- Combine related commands for each numbered part into one `bash` block. Use
  concise `# comments` inside the block to distinguish its operations.
- Mark optional major sections with `(Optional)` in the heading and state when
  they are needed. The label applies to the entire section; describe separately
  whether the playbook includes the role by default.
- Preserve stable `ops-step` IDs and `ansible_task` mappings for included manual
  operations, including grouped tasks. Validation-only tasks may remain
  Ansible-only.
- Update affected settings, operations, and verification in both paths together.
  IDs identify operations; a shared ID alone does not prove equivalent behavior.
- Use native Ansible modules where suitable.

## Ansible Layout

- Keep `ansible.cfg`, `site.yml`, `hosts.example.ini`, `group_vars/`, `host_vars/`,
  and `roles/` at the repository root.
- Use `inventory = ./hosts`; keep the real `hosts` file and local host overrides
  under `host_vars/` ignored by Git.
- Set the default SSH user to `ubuntu` in `group_vars/all.yml`.
- Keep AMD hardware and package settings in `group_vars/amd_hosts.yml` and target
  `amd_hosts`; put host-specific values in ignored host variables.
- Roles must not contain node names or host-specific package cleanup.
- Run Ansible from the repository root and include `--diff` in every
  `ansible-playbook` example or instruction.
- Keep validation in Ansible and use local playbook syntax/inventory checks.
- Do not add tests, embedded scripts, or recipe-rewriting helper scripts.

## Compatibility and Verification

- Check current AMD and Ubuntu primary sources before selecting versions.
- Distinguish architecture package availability, vendor product support,
  upstream testing, and verification on the actual host.
- A shared GFX target does not establish RX 6900 XT product or framework support.
- Require evidence before adding architecture overrides, driver replacements,
  or global library-path changes; document any necessary workaround.
- Keep package installation free of removals and driver replacements. Resolve
  existing installation conflicts separately.
- Report reboot requirements; do not reboot automatically.
- A successful package installation is not GPU or workload validation.

## Maintainer Context

Read [MAINTAINERS.md](MAINTAINERS.md) before changing recipe structure.
