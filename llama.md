# llama.cpp with Vulkan and HIP

Run Gemma 4 12B IT Q4_0 with a 4,096-token context, one request at a time and
thinking off. The default is HIP with f16/f16 cache and Flash Attention on.
See the [full benchmark table](README.md#matched-flash-attention-benchmark).

Complete [rocm.md](rocm.md), then section 1 below. Choose section 3 for HIP
(default) or section 2 for Vulkan. Run the Bash cells as your normal user; stop if a
command fails. Run from this repository directory so the HIP patch file is available.
Install `curl` with `sudo apt install curl` if it is missing.

## 1. Common setup

### 1.1 Configure the model

Set the model and service options below for manual/Runme installation:

<!-- ops-step
id: llama.settings
ansible_task: llama.settings.require | Validate build and service settings
kind: variables
-->

```bash { name=llama-settings }
export LLAMA_SOURCE=/opt/llama.cpp
export LLAMA_JOBS=8
export LLAMA_GPU_ARCH=gfx1030
export LLAMA_MODEL_ID=gemma-4-12B-it-Q4_0
export LLAMA_MODEL_REPO=ggml-org/gemma-4-12B-it-GGUF
export LLAMA_MODEL_FILE=gemma-4-12B-it-Q4_0.gguf
# export LLAMA_MMPROJ=mmproj-gemma-4-12B-it-BF16.gguf  # Optional images; BF16 only
export LLAMA_MODEL_DIR=/opt/models/$LLAMA_MODEL_ID
export LLAMA_SERVICE_USER=llama
export LLAMA_STATE_DIR=/var/lib/llama
export LLAMA_HIP_PATCH=$LLAMA_STATE_DIR/hip-rdna2-flash-attention.patch
export LLAMA_CONTEXT=4096
export LLAMA_PARALLEL=1
export LLAMA_CACHE_TYPE_K=f16
export LLAMA_CACHE_TYPE_V=f16
export LLAMA_PORT=8080
```

**Ansible alternative:** set your model in
[group_vars/amd_hosts.yml](group_vars/amd_hosts.yml):

```yaml
models:  # Models to download
  - name: gemma-4-12B-it-Q4_0
    repo: ggml-org/gemma-4-12B-it-GGUF
    file: gemma-4-12B-it-Q4_0.gguf
    # mmproj: mmproj-gemma-4-12B-it-BF16.gguf  # Optional images; BF16 only
    # mmproj_sha256: 9b1edfa05b634728ca4bfd60b4e6b278e95166c078fa54ae4fa83e680112fd1d  # Optional projector checksum
    # revision: a29b2a6b584a5bdcb672ccdcda99ea9faeacc51a  # Optional version pin
    # sha256: 3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70  # Optional checksum

llama_model: gemma-4-12B-it-Q4_0  # Name of the model to serve
```

### 1.2 Create the service user

Create the service user first and give it GPU access. Your normal user builds
llama.cpp and downloads the model; the service user reads those files.

<!-- ops-step
id: llama.user.create
ansible_task: llama.user.create | Create the service user with GPU access
kind: user
-->
<!-- ops-step
id: llama.state.prepare
ansible_task: llama.state.prepare | Prepare the service cache directory
kind: file
-->

```bash { name=llama-service-user }
if ! id "$LLAMA_SERVICE_USER" >/dev/null 2>&1; then
  sudo useradd --system --no-create-home --shell /usr/sbin/nologin "$LLAMA_SERVICE_USER"
fi
sudo usermod -aG render,video "$LLAMA_SERVICE_USER"
sudo install -d -m 755 -o "$LLAMA_SERVICE_USER" -g "$LLAMA_SERVICE_USER" "$LLAMA_STATE_DIR"
```

### 1.3 Install Hugging Face CLI

Install `hf` in a separate Python environment. `rocm.md` installs `python3-venv`.

```bash { name=llama-hf }
sudo python3 -m venv /opt/huggingface
sudo /opt/huggingface/bin/pip install huggingface_hub
sudo ln -sf /opt/huggingface/bin/hf /usr/local/bin/hf
```

Optional: run `hf auth login` without `sudo` to sign in to Hugging Face and get
higher download limits. Keep your token out of the YAML file.
The download command below enables high-performance mode to help download faster.

### 1.4 Download Gemma 4 12B

Download the text model and optional BF16 projector. Skip downloads for files
that already exist, as Ansible does. Both files use the same repository and revision.

<!-- ops-step
id: llama.model.prepare
ansible_task: llama.model.prepare | Prepare a dedicated model directory
kind: file
-->
<!-- ops-step
id: llama.model.download
ansible_task: llama.model.download | Download the model with hf
kind: command
-->

```bash { name=llama-model }
sudo install -d -m 755 -o "$USER" "$LLAMA_MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download \
  "$LLAMA_MODEL_REPO" \
  "$LLAMA_MODEL_FILE" \
  --local-dir "$LLAMA_MODEL_DIR"
chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
if [ -n "${LLAMA_MMPROJ:-}" ]; then
  HF_XET_HIGH_PERFORMANCE=1 hf download \
    "$LLAMA_MODEL_REPO" \
    "$LLAMA_MMPROJ" \
    --local-dir "$LLAMA_MODEL_DIR"
  chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MMPROJ"
fi
```

To use `curl` instead, replace only the `hf download` command with this:

```bash { name=llama-model-curl }
LLAMA_REPO_URL="https://huggingface.co/$LLAMA_MODEL_REPO"
curl -fL \
  "$LLAMA_REPO_URL/resolve/main/$LLAMA_MODEL_FILE" \
  -o "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
```

Optional download settings:

- `revision`: add `--revision VALUE` to `hf`, or replace `main` in the curl URL.
- `sha256`: run `echo 'EXPECTED_SHA256  /path/to/model.gguf' | sha256sum -c -`.
  If it fails, add `--force-download` to the download command and recheck.

Apply the same revision to both downloads. `mmproj_sha256` checks the projector
in the same way as `sha256` checks the model.

Ansible replaces files that fail the checksum. Without a checksum, it keeps existing
files. To download a different revision, remove the old file or supply its new checksum.

### 1.5 Get llama.cpp

Clone the current llama.cpp source:

<!-- ops-step
id: llama.source.prepare
ansible_task: llama.source.prepare | Prepare the source directory
kind: file
-->
<!-- ops-step
id: llama.source.checkout
ansible_task: llama.source.checkout | Check out llama.cpp
kind: command
-->

```bash { name=llama-source }
sudo install -d -m 755 -o "$USER" "$LLAMA_SOURCE"
git clone \
  --depth 1 \
  https://github.com/ggml-org/llama.cpp.git \
  "$LLAMA_SOURCE"
git -C "$LLAMA_SOURCE" rev-parse HEAD
```

For an existing clone, fetch the upstream default branch before building:

```bash { name=llama-source-update }
if git -C "$LLAMA_SOURCE" apply --reverse --check "$LLAMA_HIP_PATCH" 2>/dev/null; then
  git -C "$LLAMA_SOURCE" apply --reverse "$LLAMA_HIP_PATCH"
fi
git -C "$LLAMA_SOURCE" fetch --depth 1 origin HEAD
git -C "$LLAMA_SOURCE" checkout --detach FETCH_HEAD
git -C "$LLAMA_SOURCE" rev-parse HEAD
```

## 2. llama.cpp Vulkan

### 2.1 Build

Vulkan debug checks are off.

<!-- ops-step
id: llama.vulkan.configure
ansible_task: llama.vulkan.configure | Configure the Vulkan build
kind: command
-->
<!-- ops-step
id: llama.vulkan.build
ansible_task: llama.vulkan.build | Build the Vulkan binaries
kind: command
-->
<!-- ops-step
id: llama.backends.verify
ansible_task: llama.backends.verify | Require a GPU in the selected build
kind: verify
-->

```bash { name=llama-build-vulkan }
export LLAMA_FLASH_ATTENTION=on
cmake \
  -S "$LLAMA_SOURCE" \
  -B "$LLAMA_SOURCE/build-vulkan" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_VULKAN_CHECK_RESULTS=OFF
cmake \
  --build "$LLAMA_SOURCE/build-vulkan" \
  --target llama-server llama-cli llama-bench \
  -j "$LLAMA_JOBS"
"$LLAMA_SOURCE/build-vulkan/bin/llama-server" --list-devices
```

Check that the device list shows your GPU.

### 2.2 Start the service

Both backends use Flash Attention. Vulkan and HIP use the same service;
switching backends restarts it with the selected build.

<!-- ops-step
id: llama.systemd.write
ansible_task: llama.systemd.write | Write the selected backend service
kind: file
-->
<!-- ops-step
id: llama.systemd.start
ansible_task: llama.systemd.start | Enable and start llama.cpp
kind: command
-->

```bash { name=llama-service-vulkan }
sudo tee /etc/systemd/system/llama.service >/dev/null <<EOF
[Unit]
Description=llama.cpp - $LLAMA_MODEL_ID - vulkan
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$LLAMA_SERVICE_USER
Group=$LLAMA_SERVICE_USER
SupplementaryGroups=render video
Environment=HOME=$LLAMA_STATE_DIR
WorkingDirectory=$LLAMA_SOURCE
ExecStart=$LLAMA_SOURCE/build-vulkan/bin/llama-server \
  --model $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE \
  ${LLAMA_MMPROJ:+--mmproj $LLAMA_MODEL_DIR/$LLAMA_MMPROJ} \
  --alias $LLAMA_MODEL_ID \
  --ctx-size $LLAMA_CONTEXT \
  --parallel $LLAMA_PARALLEL \
  --flash-attn $LLAMA_FLASH_ATTENTION \
  --cache-type-k $LLAMA_CACHE_TYPE_K \
  --cache-type-v $LLAMA_CACHE_TYPE_V \
  --n-gpu-layers all \
  --host 127.0.0.1 \
  --port $LLAMA_PORT \
  --jinja \
  --reasoning-budget 0 \
  --log-verbosity 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable llama.service
sudo systemctl restart llama.service
```

### 2.3 Verify generation

Wait for `/health` to return `{"status":"ok"}`, then run the remaining checks.
The model list should show `$LLAMA_MODEL_ID`, the chat should return text, and
the logs should show all model layers loaded onto `Vulkan0` and `flash_attn = enabled`.

<!-- ops-step
id: llama.health.verify
ansible_task: llama.health.verify | Wait for the model to load
kind: verify
-->
<!-- ops-step
id: llama.models.verify
ansible_task: llama.models.verify | Verify the served model
kind: verify
-->
<!-- ops-step
id: llama.inference.verify
ansible_task: llama.inference.verify | Verify text generation
kind: verify
-->
<!-- ops-step
id: llama.offload.verify
ansible_task: llama.offload.verify | Require all model layers on the selected GPU backend
kind: verify
-->

```bash { name=llama-verify }
curl -f "http://127.0.0.1:$LLAMA_PORT/health"
curl -f "http://127.0.0.1:$LLAMA_PORT/v1/models"
curl -f \
  "http://127.0.0.1:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$LLAMA_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"temperature\":0,\"max_tokens\":32}"
sudo journalctl \
  "_SYSTEMD_INVOCATION_ID=$(systemctl show llama.service --property=InvocationID --value)" \
  --no-pager \
  -o cat
```

## 3. llama.cpp HIP

### 3.1 Build

Build HIP for RX 6900 XT (`gfx1030`) in a separate directory from Vulkan.
The included patch selects smaller attention tiles only on RDNA2 (`gfx103x`).
It keeps the occupancy assertion and enables Flash Attention for Gemma's
256/512 head sizes.

<!-- ops-step
id: llama.hip.patch
ansible_task: llama.hip.patch | Apply the RDNA2 Flash Attention tile fix
kind: command
-->

<!-- ops-step
id: llama.hip.configure
ansible_task: llama.hip.configure | Configure the HIP build
kind: command
-->
<!-- ops-step
id: llama.hip.build
ansible_task: llama.hip.build | Build the HIP binaries
kind: command
-->

```bash { name=llama-build-hip }
export LLAMA_FLASH_ATTENTION=on
if [[ "$LLAMA_GPU_ARCH" =~ ^gfx103[0-9a-f]$ ]]; then
  sudo install -m 644 roles/llama/files/hip-rdna2-flash-attention.patch "$LLAMA_HIP_PATCH"
  if ! git -C "$LLAMA_SOURCE" apply --reverse --check "$LLAMA_HIP_PATCH" 2>/dev/null; then
    git -C "$LLAMA_SOURCE" apply "$LLAMA_HIP_PATCH"
  fi
fi
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake \
  -S "$LLAMA_SOURCE" \
  -B "$LLAMA_SOURCE/build-hip" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$LLAMA_GPU_ARCH"
cmake \
  --build "$LLAMA_SOURCE/build-hip" \
  --target llama-server llama-cli llama-bench \
  -j "$LLAMA_JOBS"
"$LLAMA_SOURCE/build-hip/bin/llama-server" --list-devices
```

Check that the device list shows `ROCm0`. If the patch no longer applies after an
upstream update, stop and review it before building.

### 3.2 Start the service

Start the service with the HIP build. This also switches an existing Vulkan service to HIP.

```bash { name=llama-service-hip }
sudo tee /etc/systemd/system/llama.service >/dev/null <<EOF
[Unit]
Description=llama.cpp - $LLAMA_MODEL_ID - hip
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$LLAMA_SERVICE_USER
Group=$LLAMA_SERVICE_USER
SupplementaryGroups=render video
Environment=HOME=$LLAMA_STATE_DIR
WorkingDirectory=$LLAMA_SOURCE
ExecStart=$LLAMA_SOURCE/build-hip/bin/llama-server \
  --model $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE \
  ${LLAMA_MMPROJ:+--mmproj $LLAMA_MODEL_DIR/$LLAMA_MMPROJ} \
  --alias $LLAMA_MODEL_ID \
  --ctx-size $LLAMA_CONTEXT \
  --parallel $LLAMA_PARALLEL \
  --flash-attn $LLAMA_FLASH_ATTENTION \
  --cache-type-k $LLAMA_CACHE_TYPE_K \
  --cache-type-v $LLAMA_CACHE_TYPE_V \
  --n-gpu-layers all \
  --host 127.0.0.1 \
  --port $LLAMA_PORT \
  --jinja \
  --reasoning-budget 0 \
  --log-verbosity 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable llama.service
sudo systemctl restart llama.service
```

### 3.3 Verify generation

Run `llama-verify` from section 2.3. The logs should show all model layers loaded
onto `ROCm0` and `flash_attn = enabled`. To switch back to Vulkan, follow section 2.

## 4. Images (Optional)

### 4.1 Describe a JPEG

Use this after enabling the BF16 projector in section 1.1, downloading it and
starting either backend. For an existing installation, rerun `llama-model` and
its backend's service cell. Ansible includes the projector only when `mmproj`
is set in the selected catalog entry; rerun `llama.yml` to apply it.
Both backends passed a basic image request with this BF16 projector. When enabled,
logs should show `loaded multimodal model` and `CLIP using ROCm0` or `Vulkan0`.
The default microbatch limits Gemma's image budget to 512 tokens.

```bash { name=llama-image }
IMAGE_FILE=/path/to/image.jpg
IMAGE_DATA=$(base64 -w 0 "$IMAGE_FILE")
curl -f \
  "http://127.0.0.1:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<EOF
{
  "model": "$LLAMA_MODEL_ID",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMAGE_DATA"}},
    {"type": "text", "text": "Describe this image briefly."}
  ]}],
  "temperature": 0,
  "max_tokens": 256
}
EOF
```

The response should describe visible objects in your image. Compare it with the
image yourself; an HTTP success does not establish visual accuracy. To return
to text-only serving, unset `LLAMA_MMPROJ` and rerun the service cell, or remove
`mmproj` from the catalog and rerun Ansible.

Sources: [multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
[BF16 projector](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/main/mmproj-gemma-4-12B-it-BF16.gguf),
[build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[CMake options](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt),
[server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
