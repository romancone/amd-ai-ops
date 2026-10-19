# AMD AI Ops

Public AMD GPU recipes for Ubuntu AI hosts. The RX 6900 XT is our verification card.
Each recipe pairs readable Markdown commands with reusable Ansible roles.

## Recipes

| Recipe | Manual / Runme notebook | Ansible | Status |
|---|---|---|---|
| RX 6900 XT / Ubuntu 26.04 ROCm development libraries | [rocm.md](rocm.md) | [site.yml](site.yml) | Prepared; not deployed or workload-validated |

The recipe uses Ubuntu's `rocm-dev`, `rocminfo`, and the existing distribution
`amdgpu` driver. See [Ubuntu's installation guidance](https://ubuntu.com/blog/amd-rocm-on-ubuntu).

## Ansible

Run from the repository root. [ansible.cfg](ansible.cfg) selects `./hosts`.
Create your local inventory:

```bash
cp hosts.example.ini hosts
vim hosts
```

Real inventory and local `host_vars/` files are ignored. The default SSH user is
`ubuntu`; override it in `host_vars/<inventory_hostname>.yml` if needed:

```yaml
ansible_user: your_ssh_user
```

[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml) defines the Ubuntu release,
ROCm architecture target, and package profile. The example uses `gfx1030` for
the RX 6900 XT. Preview and apply:

```bash
ansible-playbook site.yml --check --diff
ansible-playbook site.yml --diff
```

Ansible validates the OS, loaded driver, compute-user access, and proposed APT
transaction. It installs discovery, requires the expected GPU agent, installs
the development libraries, and reminds you to reboot manually. Check mode uses existing
APT metadata and skips GPU enumeration and compiler execution. Existing package
conflicts must be resolved separately. See the guide for target prerequisites;
add `--ask-become-pass` if sudo requires a password.

Local syntax and inventory checks require no target connection:

```bash
ansible-playbook site.yml --syntax-check --diff
ansible-playbook site.yml --list-hosts --diff
```

## Manual / Runme Installation

Open [rocm.md](rocm.md) with Runme in a VS Code session connected to the target
Ubuntu host. Run cells 1.1–1.3 in order and review APT's proposal before accepting
each installation. The same commands work in a terminal.

Every cell has an explicit Runme name. With the Runme CLI installed, discover
the cells without executing them:

```bash
runme list --filename rocm.md
```

The notebook contains target-host operations only. Use the Ansible commands above
on the controller as the alternative installation path.

## Manual and Ansible Step Map

Both paths install or update `rocminfo` and `rocm-dev`, use the host's APT
recommendation settings, and verify the `gfx1030` GPU and HIP compiler. Ansible
uses `state: latest` to match `apt install`; neither path upgrades the whole system.
If changing the profile in [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml),
update the manual package names and expected GPU target to match.

| Manual / Runme part | Runme cell | Ansible operation in `roles/rocm/tasks/main.yml` |
|---|---|---|
| 1.1 Install discovery and verify GPU | `rocm-discovery` | `amd.rocm.cache.refresh`, `amd.rocm.discovery.install`, `amd.rocm.inspect.agents` |
| 1.2 Install libraries and verify compiler | `rocm-development` | `amd.rocm.install`, `amd.rocm.compiler.verify` |
| 1.3 Reboot | `rocm-reboot` | `amd.rocm.reboot.report` (manual reboot reminder) |

Stable `ops-step` comments identify the matching tasks beside each notebook cell.
Ansible additionally validates OS, settings, driver and device access, and simulates
APT changes before installing. The manual path requires its stated prerequisites
and operator review of APT's proposal and `rocminfo` output. Ansible may reuse package
metadata for up to one hour; the manual refresh always updates it.

## Compatibility and Validation

Official sources reviewed on 2026-10-20. Ubuntu 26.04 documents ROCm 7.1.0 in
Universe. [Canonical recommends](https://ubuntu.com/blog/amd-rocm-on-ubuntu)
`rocm-dev` for development libraries and headers; `rocm` provides the full suite.
The [Ubuntu package](https://packages.ubuntu.com/resolute/rocm-dev) depends on
`hipcc` and `rocminfo`. Installed versions follow Ubuntu archive updates.

Ubuntu's [hardware notes](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/#the-amd-rocm-libraries-are-now-available)
list the RX6900 family's `gfx1030` target with no CI status shown.
[AMD's current product matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
omits RX 6900 XT. Package availability does not establish vendor support or
application compatibility. GPU execution, application acceptance, and an Ansible
idempotence run remain pending.

Keep Ubuntu's existing distribution driver and package sources for this recipe.
Resolve third-party ROCm conflicts separately. Architecture overrides, driver
replacements, and global library-path changes require workload evidence.
AMD's [GPU access guidance](https://rocm.docs.amd.com/en/latest/install/rocm.html#configure-permissions-for-gpu-access)
uses `render` and `video` group membership by default; establish permissions before
starting and reconnect after group changes.

## Project Rules

See [AGENTS.md](AGENTS.md) and [MAINTAINERS.md](MAINTAINERS.md).
