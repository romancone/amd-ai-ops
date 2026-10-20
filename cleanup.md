# ROCm cleanup (Optional)

Run steps 1.1–1.4 on the host to remove an old ROCm installation.
For Ansible, use the separate example in section 1.5.

## 1. ROCm cleanup

### 1.1 Find ROCm packages

```bash { name=rocm-cleanup-review interactive=true }
# Match source packages, including libraries with lib-prefixed names.
ROCM_SOURCES='^(rocm[^ ]*|roc(minfo|blas|fft|rand|solver|sparse|thrust|prim|dbgapi|tracer|alution)|'
ROCM_SOURCES+='rocr-runtime|roct-thunk-interface|hip[^ ]*|hsa[^ ]*|miopen|rccl|comgr|amdsmi|pytorch-rocm|gloo-rocm) '
export ROCM_PACKAGES=$(
  dpkg-query -W -f='${source:Package} ${binary:Package} ${db:Status-Status}\n' |
    grep -E ' (installed|config-files)$' |
    grep -E "$ROCM_SOURCES" |
    cut -d ' ' -f2
)
printf '%s\n' "$ROCM_PACKAGES"
```

### 1.2 Preview and remove

If the list is empty, skip removal. Check APT's proposal: only ROCm packages and
dependent UCX, UCC or Open MPI packages should be removed.

```bash { name=rocm-cleanup-preview interactive=true }
apt-get --simulate purge $ROCM_PACKAGES
```

<!-- ops-step
id: amd.cleanup.packages.remove
ansible_task: amd.cleanup.packages.remove | Purge the discovered ROCm packages
kind: package
-->

```bash { name=rocm-cleanup interactive=true }
sudo apt purge $ROCM_PACKAGES
```

Do not run `autoremove` or remove kernel, GPU driver or Mesa packages.

### 1.3 Verify package removal

No selected package should have status `installed` or `config-files`.
Missing-package messages are expected.

```bash { name=rocm-cleanup-verify interactive=true }
dpkg-query -W -f='${binary:Package}: ${db:Status-Status}\n' $ROCM_PACKAGES
```

If `vulkaninfo` is installed, check that it still lists your GPU:

```bash { name=rocm-cleanup-vulkan interactive=true }
vulkaninfo --summary
```

### 1.4 Check leftover directories (Optional)

```bash { name=rocm-cleanup-directories interactive=true }
find /opt -maxdepth 1 -name 'rocm*' -print
```

Inspect any old directory before deleting files. Run `dpkg-query -S /path/to/file`
for each file; remove it with `sudo rm /path/to/file` only if no package owns it
and you no longer need it. Use `sudo rmdir /path/to/directory` for empty directories.
Do not delete a whole directory tree or follow the `/opt/rocm` symlink.

<!-- ops-step
id: amd.cleanup.roots.verify
ansible_task: amd.cleanup.roots.verify | Check for leftover legacy installation files
kind: verify
-->

Run `ls -ld /path/to/directory` for each directory you removed.
`No such file or directory` confirms removal.

Continue with [rocm.md](rocm.md) to install the new stack.

### 1.5 Ansible alternative

The role discovers installed ROCm packages and leftover package configuration.
Preview the removal plan with `ansible-playbook cleanup.yml --check --diff`, then
run `ansible-playbook cleanup.yml --diff`. No package list is required.

Optional settings in `host_vars/<inventory_hostname>.yml`:

```yaml
rocm_cleanup_roots: []  # Optional: old /opt/rocm-X.Y.Z directories to check
rocm_cleanup_files: []  # Optional: unwanted files inside those roots with no package owner
rocm_cleanup_verify_vulkan: true  # Set false if vulkaninfo is not installed
```

Run [cleanup.yml](cleanup.yml) using the [README commands](README.md#ansible).
APT may also need to remove ROCm-dependent UCX, UCC and Open MPI packages.
The role rejects other unrelated removals and preserves the GPU driver and Vulkan.
It deletes only listed unowned files and empty directories.
