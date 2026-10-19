# ROCm on Ubuntu 26.04

Install Ubuntu's ROCm development libraries using the existing `amdgpu` driver.
The verification card is an **RX 6900 XT**, with ROCm target **gfx1030**.

Requires Ubuntu 26.04 x86_64, Universe enabled, and user access to `/dev/kfd` and
the GPU render device. Run these cells on the target host as your normal user.
Review APT's proposal before accepting; stop on removals or driver/kernel changes.
Resolve any third-party ROCm conflicts separately.

This Runme notebook pairs with [site.yml](site.yml). The default settings are in
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml). See the [README](README.md)
for Ansible usage, official sources, and compatibility limits.
Status: prepared; not deployed or workload-validated.

## 1. ROCm Development Libraries

Ansible role: [rocm](roles/rocm/).

### 1.1 Install discovery and verify the GPU

Look for a GPU agent with `Name: gfx1030`. Stop if it is missing or access is denied.

<!-- ops-step
id: amd.rocm.cache.refresh
ansible_task: amd.rocm.cache.refresh | Refresh Ubuntu package metadata
kind: package
-->

<!-- ops-step
id: amd.rocm.discovery.install
ansible_task: amd.rocm.discovery.install | Install ROCm discovery without removals
kind: package
-->

<!-- ops-step
id: amd.rocm.inspect.agents
ansible_task: amd.rocm.inspect.agents | Require the expected ROCm GPU agent
kind: verify
-->

```bash { name=rocm-discovery interactive=true }
# Install discovery
sudo apt update &&
sudo apt install rocminfo &&
# Verify GPU visibility as your normal user
rocminfo
```

### 1.2 Install the development libraries and verify HIP

`rocm-dev` is a metapackage: APT installs HIP (`hipcc`, runtime and headers),
math libraries such as hipBLAS/rocBLAS and hipFFT/rocFFT, plus MIOpen and RCCL
development packages. See [Ubuntu's package list](https://packages.ubuntu.com/resolute/rocm-dev).
AI frameworks such as PyTorch are installed separately for your workload.

<!-- ops-step
id: amd.rocm.install
ansible_task: amd.rocm.install | Install the Ubuntu ROCm development profile
kind: package
-->

<!-- ops-step
id: amd.rocm.compiler.verify
ansible_task: amd.rocm.compiler.verify | Verify HIP compiler availability
kind: verify
-->

```bash { name=rocm-development interactive=true }
# Install ROCm development libraries
sudo apt install rocm-dev &&
# Verify the HIP compiler
hipcc --version
```

### 1.3 Reboot

<!-- ops-step
id: amd.rocm.reboot.report
ansible_task: amd.rocm.reboot.report | Remind the operator to reboot manually
kind: report
-->

```bash { name=rocm-reboot interactive=true }
sudo reboot
```

Reconnect and repeat `rocminfo`. Ansible prints a manual reboot reminder;
rerun the playbook after reboot to repeat its checks.

GPU discovery and compiler availability do not verify computation. Run your intended
GPU workload before calling the setup validated.
