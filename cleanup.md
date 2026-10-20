# Legacy ROCm cleanup (Optional)

Use this separate recipe when replacing a reviewed old ROCm installation.
It pairs with [cleanup.yml](cleanup.yml); the base installation never runs it automatically.
Only selected packages are purged. Ubuntu's GPU driver and Mesa stay installed.
Package purge, reviewed leftover removal, Vulkan preservation and an unchanged
second run have been verified on the RX 6900 XT host.

## 1. Legacy ROCm cleanup

Ansible role: [rocm_cleanup](roles/rocm_cleanup/).

### 1.1 Select the old installation

Record exact installed versions in ignored `host_vars/<inventory_hostname>.yml`.
The example below describes a partial ROCm 7.2 installation for Ubuntu 24.04;
change the selection to match the host. The role refuses version mismatches or
APT removals outside the selection. Defaults select nothing.

```yaml
rocm_cleanup_packages:
  rocm-core: 7.2.0.70200-43~24.04
  roctracer: 4.1.70200.70200-43~24.04
  hipblas-common-dev: 1.4.0.70200-43~24.04
  hipblaslt: 1.2.1.70200-43~24.04
  hipblaslt-dev: 1.2.1.70200-43~24.04
rocm_cleanup_roots:
  - /opt/rocm-7.2.0
rocm_cleanup_files: []
rocm_cleanup_verify_vulkan: true
```

For manual execution, inspect the same package versions and purge proposal first.
Stop if a selected version differs or APT proposes other package changes.

```bash { name=rocm-cleanup-review interactive=true }
dpkg-query -W rocm-core roctracer hipblas-common-dev hipblaslt hipblaslt-dev
apt-get --simulate purge rocm-core roctracer hipblas-common-dev hipblaslt hipblaslt-dev
```

### 1.2 Remove the reviewed packages

<!-- ops-step
id: amd.cleanup.packages.remove
ansible_task: amd.cleanup.packages.remove | Purge the reviewed legacy ROCm packages
kind: package
-->

```bash { name=rocm-cleanup interactive=true }
sudo apt purge rocm-core roctracer hipblas-common-dev hipblaslt hipblaslt-dev
```

Do not run a general autoremove. Inspect any leftover versioned directory before
removing additional files. Add only reviewed unowned regular files to
`rocm_cleanup_files`; the role rejects package-owned files and prunes only empty
directories. It stops if other files remain. For manual cleanup, use `dpkg-query -S`
to check each exact path before removing it with `sudo rm`, then remove empty
parent directories with `sudo rmdir`. These correspond to `amd.cleanup.files.remove`
and `amd.cleanup.directories.remove`; neither path deletes an unreviewed tree.

### 1.3 Verify cleanup

<!-- ops-step
id: amd.cleanup.roots.verify
ansible_task: amd.cleanup.roots.verify | Check for leftover legacy installation files
kind: verify
-->

The selected packages and `/opt/rocm-7.2.0` should be absent. Missing-package and
missing-directory messages are expected. Vulkan should still list the hardware GPU.

```bash { name=rocm-cleanup-verify interactive=true }
dpkg-query -W rocm-core roctracer hipblas-common-dev hipblaslt hipblaslt-dev
ls -ld /opt/rocm-7.2.0
vulkaninfo --summary
```

Continue with [rocm.md](rocm.md) after cleanup. This recipe does not install the new stack.
