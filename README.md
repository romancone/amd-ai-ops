# AMD AI Ops

AMD GPU infrastructure recipes for Ubuntu AI hosts, starting with the Radeon
RX 6900 XT.

This repository keeps two forms of the same operational knowledge:

- Ansible roles for repeatable, idempotent deployment.
- Markdown recipes for humans who want to read, review, and run the same steps manually.

The Markdown files work with Runme or terminal copy/paste. They keep installation
commands and essential safeguards concise; Ansible provides complete validation,
runtime checks, and check-mode handling.

## Current Recipes

| Recipe | Human doc | Automation | Status |
|---|---|---|---|
| Ubuntu ROCm with the distribution AMDGPU driver | [rocm.md](rocm.md) | [site.yml](site.yml) | Prepared; not deployed or GPU-validated |

## Ansible Quick Start

Run from the repository root. [ansible.cfg](ansible.cfg) selects the single local
inventory:

```ini
[defaults]
inventory = ./hosts
```

On a new checkout, create the local inventory from the example:

```bash
cp hosts.example.ini hosts
vim hosts
```

`hosts` and local files under `host_vars/` are gitignored. Keep real hostnames,
connection details, and reviewed legacy-package migration settings there.
[group_vars/all.yml](group_vars/all.yml) sets `ansible_user: ubuntu`; override it
in `host_vars/<inventory_hostname>.yml` when needed:

```yaml
ansible_user: your_ssh_user
```

AMD hardware and package settings live in
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml). The initial profile is Ubuntu
26.04 x86_64 and RX 6900 XT (`1002:73bf`, `gfx1030`). The playbook preserves the
distribution `amdgpu` driver, verifies discovery, reviews legacy cleanup, installs
Ubuntu's `rocm-dev`, and runs a bounded HIP test. It installs no inference framework.

After deployment authorization, start with discovery and review its result before
the complete migration:

```bash
ansible-playbook site.yml --tags rocm_discovery --diff
ansible-playbook site.yml --diff
```

Both commands can change the target. See [rocm.md](rocm.md) for prerequisites,
sudo requirements, check-mode limits, acceptance, and recovery.

Local validation needs no target connection:

```bash
ansible-playbook site.yml --syntax-check --diff
ansible-playbook site.yml --list-hosts --diff
python3 -m unittest discover -s tests -v
```

## Human Recipe Quick Start

Open [rocm.md](rocm.md), review the settings, and run each numbered logical part
in order on the target host. The three top-level sections match the playbook role
order. Related commands share one shell block with concise operation comments.
Use one Bash session, or a shared Runme shell session, to preserve variables.

## Recipe Alignment

Included manual operations use stable IDs matching Ansible task names, for example:

```text
amd.rocm.discovery.install
amd.rocm.inspect.agents
amd.rocm.legacy.remove
amd.rocm.install
amd.rocm.smoke.run
```

The paths share settings, installation order, safeguards, and expected results:

- manual recipes keep installation commands and final verification concise
- Ansible roles include full validation, runtime checks, and check-mode handling
- included operations map to Ansible tasks, including grouped commands;
  validation-only tasks may remain Ansible-only

Both implementations are maintained directly. Stable IDs and offline contract
tests support review; they do not prove full semantic equivalence or GPU compatibility.

## Compatibility and Validation Status

The baseline uses Ubuntu's native `rocm-dev` and existing kernel driver.
[Canonical documents this package path](https://ubuntu.com/blog/amd-rocm-on-ubuntu);
[Ubuntu's release notes](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/#the-amd-rocm-libraries-are-now-available)
list the RX6900 family's `gfx1030` target with a blank CI-status cell.

Package availability, vendor product support, and actual host validation remain
separate. Deployment, HIP execution, application acceptance, and idempotence
verification are pending.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).
