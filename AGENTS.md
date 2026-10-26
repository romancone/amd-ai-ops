# Project Rules

## Scope and Review
- Stay within the requested change. Do not add models, frameworks, benchmarks,
  helper scripts or extra test repetitions without a request
- Treat review, tests, host changes, commits, pushes and merges as separate actions.
  Respect explicit prohibitions across turns until the user changes them.
  "No tests" includes syntax checks and benchmark reruns; use static inspection
- Keep review changes unstaged when requested. Inspect the current working diff;
  do not mix it with previously committed or parked work
- Keep raw run logs and temporary evidence outside the repository. Approved
  fixtures, scoring references and condensed reports may be tracked in `media/`,
  `evaluation/` and `reports/`. Keep result tables in BENCHMARK.md.
  Preserve earlier results and distinguish completed, failed and pending runs

## Git

- Use `codex/<topic>` feature branches for new work. If a branch change has not
  already been requested, propose its name for human confirmation before creating it
- Never commit or push without explicit human authorization
- Continue on the current branch when instructed; preserve unrelated pending work
- Keep commits reviewable and scoped to one feature or cleanup

## Writing Style

- Do not end bullet items with a period, in Markdown or HTML
- **Write in clean, simple language and keep it concise.** Apply this to every
  guide section and user-facing explanation
- Manual recipes must be usable on their own: runnable commands with short comments.
  Do not require Ansible inventory files to follow manual steps. Put Ansible
  configuration in a separate, clearly labelled example
- Show the editable configuration example and its file path before commands that
  depend on it. A description of the roles is not a substitute for configuration
- Explain parameters beside their values using concise YAML or Bash `#` comments.
  Show optional parameters commented out. Do not repeat those explanations in prose
- Remove repeated reminders, vague "review/update as needed" instructions, internal
  implementation details and development history from operator steps
- State what to run and what output confirms success. Keep essential conditions
  clear without turning a short instruction into a long disclaimer
- Do not use ASCII decoration or verbose comments

## Command Formatting

- Use standard Ubuntu commands. Omit redundant flags and checks
- Put separate commands on separate lines; do not chain them with `&&`
- Keep the command and short flags together: `curl -fL \`, not a bare `curl`
  followed by a separate line containing only `-fL`
- Wrap long commands or long arguments with `\` at logical boundaries.
  Put each CMake definition or server option with its value on its own continuation line. Do not split every short argument mechanically or leave the entire command on one long line
- When a recipe requires a reboot, show `sudo reboot`; do not add a shell check
  just to decide whether to print or run it. Never reboot a host without authorization

## Recipes

- Keep Ansible roles reusable and host-agnostic
- Roles define installation and service behavior; inventory supplies model and
  host values. Recipes, configuration examples and the Step Map derive from them
- Align manual recipes with their Ansible roles in settings, installation order,
  model paths, build flags, service arguments and final result. A prose-only edit
  should leave commands unchanged unless a correction requires changing them
- Keep manual recipes simple. Skip redundant variable, OS, and preflight checks,
  change tracking, and check-mode reports. Retain essential safeguards and final
  verification
- Ansible roles must include complete prerequisite and settings validation,
  runtime checks, idempotence, and check-mode handling
- Use two-level numbering in Markdown recipes: `## 1. Role name`, then
  `### 1.1 Logical part`, `### 1.2 Logical part`, and so on
- Keep generic libraries and AMD-specific libraries in separate sections and blocks
- Combine related commands for each numbered part into one `bash` block. Use
  named Runme cells and concise `# comments` to distinguish operations
- Mark optional major sections with `(Optional)` in the heading and state when
  they are needed. The label applies to the entire section; describe separately
  whether the playbook includes the role by default
- Preserve stable `ops-step` IDs and `ansible_task` mappings for included manual
  operations, including grouped tasks. Validation-only tasks may remain
  Ansible-only
- Update affected settings, operations, and verification in both paths together.
  IDs identify operations; a shared ID alone does not prove equivalent behavior
- Use native Ansible modules where suitable

## Ansible Layout

- Keep `ansible.cfg`, `site.yml`, `hosts.example.ini`, `group_vars/`, `host_vars/`,
  and `roles/` at the repository root
- Use `inventory = ./hosts`; keep the real `hosts` file and local host overrides
  under `host_vars/` ignored by Git
- Set the default SSH user to `ubuntu` in `group_vars/all.yml`
- Keep AMD hardware and package settings in `group_vars/amd_hosts.yml` and target
  `amd_hosts`; put host-specific values in ignored host variables
- Keep model configuration in inventory
- Keep the model catalog independent of the inference engine. Roles must support
  the list without embedding a particular model in their defaults. Populate one
  model and use that same example throughout the manual until asked to add more
- Model `revision` and `sha256` are optional. Show them as commented-out fields;
  basic installation must work without either. Use clear names, not redundant
  combinations such as `model_store_model`
- Roles must not contain node names or host-specific package cleanup
- Run Ansible from the repository root and include `--diff` in every
  `ansible-playbook` example or instruction
- Keep validation in Ansible. Run local syntax/inventory checks only when testing
  is within the current scope; never override a user's "no tests" instruction
- Do not add tests, embedded scripts, or recipe-rewriting helper scripts

## Compatibility and Verification

- Public recipes and role defaults must not require hardcoded version pins.
  Discover installed versions or omit pinning; optional model pins remain opt-in.
  Cleanup takes package names, not a package/version map. Discover actual leftover
  directory paths rather than copying a versioned path from the verification host
- Check current AMD and Ubuntu primary sources before selecting versions
- This is a public, reusable developer repository. Treat the RX 6900 XT as the
  verification card, not a host identity to embed in roles. Use `gfx1030` where
  compilation needs it; omit unnecessary PCI IDs from the installation guide
- Distinguish architecture package availability, vendor product support,
  upstream testing, and verification on the actual host
- A shared GFX target does not establish RX 6900 XT product or framework support
- Require evidence before adding architecture overrides, driver replacements,
  or global library-path changes; document any necessary workaround
- Keep package installation free of removals and driver replacements. Resolve
  existing installation conflicts separately
- Report reboot requirements; do not reboot automatically
- A successful package installation is not GPU or workload validation
- When tests are requested, report both requested backends and the actual model,
  cache, concurrency and relevant results. Respect the requested test combinations;
  do not silently substitute another quantization or expand the matrix
- Keep recommendations separate from measurements. State the recommended command
  and when to use it; never present an untested configuration as verified
- Retry HIP Flash Attention failures with FA off and retain both results.
  Quantized V caches require FA; record rejected combinations as unsupported,
  and use a clearly labelled f16/f16 fallback. Do not replace the original settings silently

## Maintainer Context

Read [MAINTAINERS.md](MAINTAINERS.md) before changing recipe structure.

## Temporary Internal Evidence

Temporary review notes, not operator instructions or a claim of current host state.

- Matrix collected on 2026-10-22: 54 configurations, 108 pp/tg cases,
  324 benchmark samples and 216 image answers at 16K, with FA on
- Matrix logs: `../amd-ai-ops-evidence-2026-10-22/vision-20261021T221846Z`
- Corrected Qwen HIP logs:
  `../amd-ai-ops-evidence-2026-10-22/qwen-fa-fix-20261022/node2/h3`
- Service checks on 2026-10-23 passed for both backends at 16K: model identity,
  text generation, BF16 projector placement, FA on and all 49 layers on GPU.
  These were service checks, not new pp/tg or four-image measurements
- Service-check logs:
  `../amd-ai-ops-evidence-2026-10-24/session-evidence/amd-context16k-20261023`
- External evidence index: `../amd-ai-ops-evidence-2026-10-24/README.md`
- Q8 maximum-context results: 10 passes, 4 failures, 4 memory exclusions;
  14 pp/tg pairs (84 samples), 52 successful image answers and 4 image errors.
  Successful answers match the 16K answers and use the same Luna-6 scores
- Maximum-context logs, corrected token counts and restored service state:
  `../amd-ai-ops-evidence-2026-10-24/q8-full-context/`.
  Token-count corrections are recorded as `validation_correction`

### Parked Next-Step Evaluation

Pending; do not run until explicitly requested:

1. Test MTP on versus off with supported model artifacts
2. Test 2 and 4 concurrent slots
3. Compare unified KV cache on/off and cache capacity at those concurrency levels

All 54 16K configurations and the maximum-context checks used one slot with
`kv_unified = false`. They do not establish capacity for multiple users.
