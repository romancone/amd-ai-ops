# Vulkan and HIP development on Ubuntu 26.04

- **Build tools and Python** provide C/C++ compilation, build systems and isolated Python environments.
- **Vulkan** provides portable GPU compute through Mesa, with headers and shader compilation tools.
- **HIP / ROCm** provides AMD's GPU compiler, runtime, math and machine-learning development libraries.
- **ROCm PyTorch** provides AMD GPU tensors and training through Ubuntu's system Python.
- **Legacy cleanup (optional)** removes a reviewed old ROCm installation before setting up the new stack; see [cleanup.md](cleanup.md).

For Ubuntu 26.04 x86_64 with Universe enabled, the existing Ubuntu `amdgpu`
driver, and user access to `/dev/kfd` and the GPU render device. The verification
card is RX 6900 XT (`gfx1030`). Run as your normal user.

This Runme notebook pairs with [site.yml](site.yml). Review APT's proposal before
accepting; resolve removals, kernel changes and third-party ROCm conflicts separately.
Status: installed on the verification card; Vulkan/HIP compute passed. Basic PyTorch
training passed with the process-local workaround in section 2.3.

## 1. Generic development libraries

Ansible role: [development](roles/development/). These packages are vendor-neutral.
APT keeps already-current packages; existing build tools do not need reinstalling.

### 1.1 Build tools and Python

<!-- ops-step
id: development.cache.refresh
ansible_task: development.cache.refresh | Refresh Ubuntu package metadata
kind: package
-->
<!-- ops-step
id: development.install
ansible_task: development.install | Install generic development tools
kind: package
-->

```bash { name=development-tools interactive=true }
sudo apt update
sudo apt install build-essential cmake ninja-build git pkgconf libssl-dev python3-dev python3-venv
```

### 1.2 Vulkan

Vulkan headers and shader tools support compilation. Mesa provides the Vulkan
driver, including RADV for AMD. Expect the actual GPU in `vulkaninfo`, not only llvmpipe.

<!-- ops-step
id: development.vulkan.install
ansible_task: development.vulkan.install | Install generic Vulkan development and Mesa packages
kind: package
-->
<!-- ops-step
id: development.vulkan.verify
ansible_task: development.vulkan.verify | Require a hardware Vulkan device
kind: verify
-->

```bash { name=vulkan-development interactive=true }
sudo apt install libvulkan-dev glslc spirv-headers mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary
```

## 2. AMD development libraries

Ansible role: [rocm](roles/rocm/). Settings: [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml).

### 2.1 GPU discovery

Expect a GPU agent named `gfx1030`. Stop if it is missing or access is denied.

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
sudo apt update
sudo apt install rocminfo
rocminfo
```

### 2.2 HIP compiler and libraries

`rocm-dev` installs HIP, headers, hipBLAS/rocBLAS, MIOpen, RCCL and other
development libraries. `rocm-smi` provides monitoring.

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
sudo apt install rocm-dev rocm-smi
hipcc --version
```

### 2.3 AMD PyTorch for Python

Use Ubuntu's ROCm-enabled PyTorch with the system Python. Its dependencies include
NumPy and the required ROCm runtime libraries. Do not replace it with an unrelated
`pip install torch` in the same environment.

The tested Ubuntu PyTorch 2.9.1 package needs AMD's compiler library loaded first
for GPU work. Use the `LD_PRELOAD` prefix below for each PyTorch command; it applies
only to that process. Ansible uses the matching `rocm_python_preload` setting.
See the [verification notes](README.md#compatibility-and-acceptance).

<!-- ops-step
id: amd.rocm.python.install
ansible_task: amd.rocm.python.install | Install Ubuntu ROCm PyTorch
kind: package
-->
<!-- ops-step
id: amd.rocm.python.inspect
ansible_task: amd.rocm.python.inspect | Report the PyTorch environment
kind: verify
-->

```bash { name=rocm-python interactive=true }
sudo apt install python3-torch-rocm
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libamd_comgr.so.3 /usr/bin/python3 -m torch.utils.collect_env
```

### 2.4 Reboot

<!-- ops-step
id: amd.rocm.reboot.report
ansible_task: amd.rocm.reboot.report | Remind the operator to reboot manually
kind: report
-->

```bash { name=rocm-reboot interactive=true }
sudo reboot
```

Reconnect and repeat `vulkaninfo --summary`, `rocminfo` and the PyTorch environment
report. Ansible prints a reboot reminder; it does not reboot the host.
Application builds remain outside this recipe. An environment report alone does
not verify computation; run the intended workload with the same Python prefix.

Sources: [Ubuntu ROCm development packages](https://packages.ubuntu.com/resolute/rocm-dev),
[Ubuntu ROCm PyTorch](https://packages.ubuntu.com/resolute/python3-torch-rocm),
[llama.cpp Vulkan and HIP build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
