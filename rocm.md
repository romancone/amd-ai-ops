# Ubuntu ROCm with the Distribution AMDGPU Driver

- **Distribution AMDGPU driver** provides GPU and KFD access through Ubuntu's kernel.
- **ROCm development profile** provides discovery, the HIP compiler, and compute libraries.
- **HIP smoke test** compiles and verifies a small GPU computation.

This is the human-readable pair for [site.yml](site.yml). Status: **prepared,
not deployed or GPU-validated**. The initial profile targets Ubuntu 26.04 x86_64
and RX 6900 XT (`1002:73bf`, `gfx1030`) with Universe enabled. The driver and
compute-user access must already work; this recipe does not replace drivers,
configure repositories, change groups, or reboot the host.

Run manual blocks in order on the target as the intended non-root compute user
in one Bash session, or a shared Runme shell session. Requires `python3`,
`python3-apt`, `pciutils`, and `kmod`. Use sudo only for package operations.
Ansible performs complete settings, OS, runtime, and check-mode validation.

Run Ansible from the repository root; `ansible.cfg` selects the ignored `hosts`
inventory. Local host overrides contain the actual SSH user and cleanup list.
After deployment authorization, first run discovery, review its result, then
run the full migration:

```bash
ansible-playbook site.yml --tags rocm_discovery --diff
ansible-playbook site.yml --diff
```

APT tasks need sudo access; add `--ask-become-pass` if required. Check mode uses
existing package metadata, reports deferred discovery if its tool is absent,
and skips HIP execution; it is not GPU acceptance:

```bash
ansible-playbook site.yml --check --diff
```

## 1. Distribution AMDGPU Driver

Ansible role: [amdgpu_preflight](roles/amdgpu_preflight/).

### 1.1 Configure the hardware profile

```bash
# Configure the existing GPU; run all manual parts in this Bash session
set -euo pipefail
export AMD_GPU_PCI_ID="1002:73bf"
```

### 1.2 Verify the existing driver and compute access

Keep the distribution driver. Stop if the GPU binding, module path, or device
access check fails; review the host instead of adding driver or permission changes.

<!-- ops-step
id: amd.driver.inspect.pci
ansible_task: amd.driver.inspect.pci | Require the expected GPU on amdgpu
kind: check
-->

<!-- ops-step
id: amd.driver.inspect.module
ansible_task: amd.driver.inspect.module | Require the distribution kernel module
kind: check
-->

<!-- ops-step
id: amd.driver.verify.access
ansible_task: amd.driver.verify.access | Require compute-user access to the matching GPU
kind: verify
-->

```bash
# Inspect the GPU binding
pci_state="$(LC_ALL=C lspci -nnk -d "$AMD_GPU_PCI_ID")"
printf '%s\n' "$pci_state"
grep -Fq 'Kernel driver in use: amdgpu' <<< "$pci_state"

# Require the distribution module
module_path="$(modinfo -F filename amdgpu)"
printf '%s\n' "$module_path"
[[ "$module_path" == */kernel/drivers/gpu/drm/amd/amdgpu/* ]]
[[ "$module_path" != */updates/* ]]

# Check access to this GPU as the compute user
python3 - "$AMD_GPU_PCI_ID" <<'PY'
from pathlib import Path
import os
import sys

vendor, device = (int(value, 16) for value in sys.argv[1].split(':'))
nodes = []
for render in Path('/sys/class/drm').glob('renderD*'):
    pci = render / 'device'
    if (int((pci / 'vendor').read_text(), 16),
            int((pci / 'device').read_text(), 16)) == (vendor, device):
        nodes.append(Path('/dev/dri') / render.name)
if not nodes:
    raise SystemExit('No render device matches the configured GPU')
for node in [Path('/dev/kfd')] + nodes:
    if not os.access(node, os.R_OK | os.W_OK):
        raise SystemExit(f'Compute user needs read/write access to {node}')
    print(f'Accessible: {node}')
PY
```

## 2. ROCm Development Profile

Ansible role: [rocm](roles/rocm/).

### 2.1 Configure packages and the GPU target

These cleanup names and the version pattern describe the inspected legacy
Ubuntu 24.04 build. On a clean host, set `ROCM_LEGACY_PACKAGES` to an empty string.
Ansible's cleanup list is empty by default and configured per host.

```bash
# Configure ROCm settings
export ROCM_GFX_TARGET="gfx1030"
export ROCM_PROFILE_PACKAGES="rocm-dev"
export ROCM_LEGACY_PACKAGES="rocm-core roctracer hipblas-common-dev hipblaslt hipblaslt-dev"
export ROCM_LEGACY_VERSION_PATTERN='70200.*~24\.04$'
```

### 2.2 Install discovery and verify the GPU agent

Refresh metadata, install discovery without removals, and require the expected
GPU agent. Stop here for the discovery-only first pass.

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

```bash
# Refresh Ubuntu package metadata
sudo apt-get update

# Install discovery without removals
sudo apt-get --no-remove -o APT::Install-Recommends=true install rocminfo

# Verify the configured GPU agent
agents="$(LC_ALL=C /usr/bin/rocminfo)"
printf '%s\n' "$agents"
grep -Eq "^[[:space:]]*Name:[[:space:]]+$ROCM_GFX_TARGET[[:space:]]*$" <<< "$agents"
```

### 2.3 Review the migration transaction

Keep these essential removal safeguards. Reject mismatched legacy versions,
unrelated removals, installed-package upgrades, kernel/driver changes, or
non-Ubuntu package origins. Review the printed plan before the next part.

<!-- ops-step
id: amd.rocm.legacy.validate
ansible_task: amd.rocm.legacy.validate | Require the expected legacy package versions
kind: check
-->

<!-- ops-step
id: amd.rocm.transaction.inspect
ansible_task: amd.rocm.transaction.inspect | Simulate the complete ROCm migration
kind: check
-->

<!-- ops-step
id: amd.rocm.transaction.validate
ansible_task: amd.rocm.transaction.validate | Reject changes outside the reviewed migration
kind: check
-->

```bash
# Select only installed legacy packages with the reviewed build version
ROCM_LEGACY_PRESENT="$(python3 - <<'LEGACY'
import os, re, subprocess
present = []
for package in os.environ['ROCM_LEGACY_PACKAGES'].split():
    result = subprocess.run(['dpkg-query', '-W', '-f=${db:Status-Abbrev} ${Version}', package],
                            text=True, capture_output=True)
    fields = result.stdout.split(None, 1)
    if result.returncode == 0 and len(fields) == 2 and fields[0] == 'ii':
        assert re.search(os.environ['ROCM_LEGACY_VERSION_PATTERN'], fields[1]), package
        present.append(package)
print(' '.join(sorted(present)))
LEGACY
)"
export ROCM_LEGACY_PRESENT

# Simulate the complete migration before removing anything
read -r -a profile_packages <<< "$ROCM_PROFILE_PACKAGES"
legacy_arguments=()
for package in $ROCM_LEGACY_PRESENT; do legacy_arguments+=("$package-"); done
ROCM_TRANSACTION="$(LC_ALL=C apt-get --simulate -o Debug::NoLocking=1 \
  -o APT::Install-Recommends=true install "${profile_packages[@]}" "${legacy_arguments[@]}")"
export ROCM_TRANSACTION
printf '%s\n' "$ROCM_TRANSACTION"

# Reject changes outside the reviewed migration
python3 - <<'PLAN'
import os, re
plan = os.environ['ROCM_TRANSACTION']
allowed = set(os.environ['ROCM_LEGACY_PACKAGES'].split())
assert not (set(re.findall(r'^Remv ([^ ]+)', plan, re.M)) - allowed)
assert not re.search(r'^Inst (amdgpu-dkms|linux-(image|headers|modules|generic))', plan, re.M)
assert not re.search(r'^Inst \S+ \[', plan, re.M)
assert all('Ubuntu:' in line for line in re.findall(r'^Inst .+$', plan, re.M))
PLAN
```

### 2.4 Remove the reviewed remnants and install ROCm

Remove only the reviewed installed packages. Keep configuration files; do not
purge, autoremove, or manually delete `/opt`.

<!-- ops-step
id: amd.rocm.legacy.remove
ansible_task: amd.rocm.legacy.remove | Remove only the reviewed legacy packages
kind: package
-->

<!-- ops-step
id: amd.rocm.install
ansible_task: amd.rocm.install | Install the Ubuntu ROCm development profile
kind: package
-->

```bash
# Remove only the reviewed legacy packages
if [ -n "$ROCM_LEGACY_PRESENT" ]; then
    read -r -a legacy_packages <<< "$ROCM_LEGACY_PRESENT"
    sudo apt-get -o APT::Get::AutomaticRemove=false remove "${legacy_packages[@]}"
fi

# Install the Ubuntu profile without additional removals
read -r -a profile_packages <<< "$ROCM_PROFILE_PACKAGES"
sudo apt-get --no-remove -o APT::Install-Recommends=true install "${profile_packages[@]}"
```

### 2.5 Verify installed package versions

Keep these versions with the deployment results.

<!-- ops-step
id: amd.rocm.versions.report
ansible_task: amd.rocm.versions.report | Report installed ROCm package versions
kind: check
-->

<!-- ops-step
id: amd.rocm.versions.show
ansible_task: amd.rocm.versions.show | Show the installed package versions
kind: report
-->

```bash
# Report the installed discovery, compiler, and profile versions
read -r -a profile_packages <<< "$ROCM_PROFILE_PACKAGES"
dpkg-query -W -f='${binary:Package} ${Version}\n' rocminfo hipcc "${profile_packages[@]}"
```

## 3. HIP Compute Verification

Ansible role: [rocm_smoke_test](roles/rocm_smoke_test/).

The full playbook includes this role by default. Its GPU test is skipped in check
mode; an actual run is required to validate compute.

### 3.1 Compile and verify GPU computation

Create a temporary test, select a HIP device with the configured target, and
verify 1024 results. Compilation is limited to 120 seconds and execution to 60.
The inline source matches the role file; the EXIT trap cleans up on failure.

<!-- ops-step
id: amd.rocm.smoke.workspace
ansible_task: amd.rocm.smoke.workspace | Create a temporary compute-test directory
kind: file
-->

<!-- ops-step
id: amd.rocm.smoke.source
ansible_task: amd.rocm.smoke.source | Write the HIP compute test
kind: file
-->

<!-- ops-step
id: amd.rocm.smoke.compile
ansible_task: amd.rocm.smoke.compile | Compile for the configured GPU architecture
kind: command
-->

<!-- ops-step
id: amd.rocm.smoke.run
ansible_task: amd.rocm.smoke.run | Execute and verify the HIP kernel
kind: verify
-->

<!-- ops-step
id: amd.rocm.smoke.report
ansible_task: amd.rocm.smoke.report | Report the verified compute result
kind: report
-->

<!-- ops-step
id: amd.rocm.smoke.cleanup
ansible_task: amd.rocm.smoke.cleanup | Remove temporary compute-test files
kind: file
-->

```bash
# Create the temporary workspace and arrange cleanup
ROCM_SMOKE_DIR="$(mktemp -d /tmp/rocm-smoke-XXXXXX)"
export ROCM_SMOKE_DIR
trap 'rm -rf -- "$ROCM_SMOKE_DIR"' EXIT

# Write the HIP compute test
cat > "$ROCM_SMOKE_DIR/vector_add.cpp" <<'CPP'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

#define HIP_OK(call) do { \
    hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
        return 1; \
    } \
} while (false)

__global__ void add_one(float* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += 1.0f;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string expected = argv[1];
    int count = 0;
    HIP_OK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_OK(hipGetDeviceProperties(&properties, device));
        std::string target = properties.gcnArchName;
        if (target.substr(0, target.find(':')) == expected) {
            selected = device;
            break;
        }
    }
    if (selected < 0) {
        std::fprintf(stderr, "No HIP device matches %s\n", expected.c_str());
        return 1;
    }
    HIP_OK(hipSetDevice(selected));
    constexpr int elements = 1024;
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) values[index] = float(index);
    float* device_values = nullptr;
    HIP_OK(hipMalloc(reinterpret_cast<void**>(&device_values), elements * sizeof(float)));
    HIP_OK(hipMemcpy(device_values, values.data(), elements * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(add_one, dim3(4), dim3(256), 0, 0, device_values, elements);
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(values.data(), device_values, elements * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipFree(device_values));
    for (int index = 0; index < elements; ++index) {
        if (values[index] != float(index + 1)) {
            std::fprintf(stderr, "Incorrect result at %d\n", index);
            return 1;
        }
    }
    std::printf("PASS target=%s device=%d elements=%d\n", expected.c_str(), selected, elements);
    return 0;
}
CPP

# Compile for the configured GPU
timeout 120 /usr/bin/hipcc -std=c++17 --offload-arch="$ROCM_GFX_TARGET" \
  "$ROCM_SMOKE_DIR/vector_add.cpp" -o "$ROCM_SMOKE_DIR/vector_add"

# Run the test and print its PASS result
timeout 60 "$ROCM_SMOKE_DIR/vector_add" "$ROCM_GFX_TARGET"

# Remove temporary files
rm -rf -- "$ROCM_SMOKE_DIR"
trap - EXIT
```

### 3.2 Accept the results and handle failures

Expect `PASS target=gfx1030 ... elements=1024` for the initial profile. This proves
only the small HIP computation, not an inference framework. Record OS, kernel,
GPU identity, package versions, and command results. The intended application
workload and a deployment idempotence run are still required for acceptance.

No reboot is planned because the kernel driver is preserved. If one becomes
necessary, stop and report it to the operator. Ansible's package-cache refresh
may report a change after 3600 seconds; temporary smoke-test files leave no
persistent configuration and report unchanged.

Cleanup and profile installation are separate APT transactions. If installation
fails after cleanup, resolve the APT error and rerun. Restoring legacy packages
requires their original archives or repository and is not automated. Do not
manually delete directories or add architecture/global library-path overrides.

References: [Ubuntu ROCm packages](https://ubuntu.com/blog/amd-rocm-on-ubuntu),
[Ubuntu hardware notes](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/#the-amd-rocm-libraries-are-now-available),
and [HIP kernel launch API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html).
