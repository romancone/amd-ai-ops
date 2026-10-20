# llama.cpp with Vulkan and HIP

A native systemd service for Gemma 4 12B instruction-tuned text generation.
Build both backends, then select one for the service. Start with Q4_0 weights,
a 4,096-token context and one request slot. Thinking is disabled for this baseline.

Complete [rocm.md](rocm.md) first: generic build/Vulkan libraries and AMD HIP
libraries stay in their separate sections there. This recipe also requires `curl`
(`sudo apt install curl` if it is missing).
Run the named Bash cells in order as your normal user. Stop on any failed command.
Ansible equivalent: [llama.yml](llama.yml), role [llama](roles/llama/).

## 1. llama.cpp service

### 1.1 Settings

These defaults match [the role](roles/llama/defaults/main.yml). The service binds
to localhost. Use `hip` instead of `vulkan` to select the AMD backend.

<!-- ops-step
id: llama.settings
ansible_task: llama.settings.require | Validate build and service settings
kind: variables
-->

```bash { name=llama-settings }
export LLAMA_SOURCE=/opt/llama.cpp
export LLAMA_BACKEND=vulkan
export LLAMA_GPU_ARCH=gfx1030
export LLAMA_JOBS=8
export LLAMA_MODEL_DIR=/opt/models/gemma-4-12B-it-Q4_0
export LLAMA_MODEL_FILE=gemma-4-12B-it-Q4_0.gguf
export LLAMA_MODEL_NAME=gemma-4-12B-it
export LLAMA_MODEL_URL=https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/a29b2a6b584a5bdcb672ccdcda99ea9faeacc51a/gemma-4-12B-it-Q4_0.gguf
export LLAMA_MODEL_SHA256=3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70
export LLAMA_SERVICE_USER=llama
export LLAMA_STATE_DIR=/var/lib/llama
export LLAMA_CONTEXT=4096
export LLAMA_FLASH_ATTENTION=off
export LLAMA_PORT=8080
```

### 1.2 Get llama.cpp

Pinned release: `v0.7.0`, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
On an existing checkout, fetch and check out this release instead of cloning again.

<!-- ops-step
id: llama.source.prepare
ansible_task: llama.source.prepare | Prepare the source directory
kind: file
-->
<!-- ops-step
id: llama.source.checkout
ansible_task: llama.source.checkout | Check out the pinned llama.cpp release
kind: command
-->

```bash { name=llama-source }
sudo mkdir -p "$LLAMA_SOURCE"
sudo chown "$USER" "$LLAMA_SOURCE"
git clone --branch v0.7.0 --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SOURCE"
git -C "$LLAMA_SOURCE" rev-parse HEAD
```

### 1.3 Build Vulkan

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

```bash { name=llama-build-vulkan }
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_SOURCE/build-vulkan" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build "$LLAMA_SOURCE/build-vulkan" --target llama-server llama-cli llama-bench -j "$LLAMA_JOBS"
```

### 1.4 Build HIP

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
<!-- ops-step
id: llama.backends.verify
ansible_task: llama.backends.verify | Require a GPU in each build
kind: verify
-->

```bash { name=llama-build-hip }
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S "$LLAMA_SOURCE" -B "$LLAMA_SOURCE/build-hip" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS="$LLAMA_GPU_ARCH"
cmake --build "$LLAMA_SOURCE/build-hip" --target llama-server llama-cli llama-bench -j "$LLAMA_JOBS"
"$LLAMA_SOURCE/build-vulkan/bin/llama-server" --list-devices
"$LLAMA_SOURCE/build-hip/bin/llama-server" --list-devices
```

Both device lists must show the real GPU. CPU-only output is not a successful build.

### 1.5 Download Gemma 4 12B

The Q4_0 file is 7.22 GB. Download only the text model for this first test.
Keep existing models in their own directories.

<!-- ops-step
id: llama.model.prepare
ansible_task: llama.model.prepare | Prepare a dedicated model directory
kind: file
-->
<!-- ops-step
id: llama.model.download
ansible_task: llama.model.download | Download and verify the pinned model
kind: command
-->

```bash { name=llama-model }
sudo mkdir -p "$LLAMA_MODEL_DIR"
sudo chown "$USER" "$LLAMA_MODEL_DIR"
curl -fL "$LLAMA_MODEL_URL" -o "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
echo "$LLAMA_MODEL_SHA256  $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE" | sha256sum -c -
chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
```

### 1.6 Create the service user

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

### 1.7 Create and start the service

Flash Attention is disabled in this baseline: the HIP path hit a kernel-occupancy
assertion on the verification card with Gemma 4. Log verbosity 4 records GPU placement.

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

```bash { name=llama-service }
sudo tee /etc/systemd/system/llama.service >/dev/null <<EOF
[Unit]
Description=llama.cpp - $LLAMA_MODEL_NAME - $LLAMA_BACKEND
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$LLAMA_SERVICE_USER
Group=$LLAMA_SERVICE_USER
SupplementaryGroups=render video
Environment=HOME=$LLAMA_STATE_DIR
WorkingDirectory=$LLAMA_SOURCE
ExecStart=$LLAMA_SOURCE/build-$LLAMA_BACKEND/bin/llama-server --model $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE --alias $LLAMA_MODEL_NAME --ctx-size $LLAMA_CONTEXT --parallel 1 --flash-attn $LLAMA_FLASH_ATTENTION --n-gpu-layers all --host 127.0.0.1 --port $LLAMA_PORT --jinja --reasoning-budget 0 --log-verbosity 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable llama.service
sudo systemctl restart llama.service
```

### 1.8 Verify generation

Wait for `/health` to return `{"status":"ok"}`, then run the model and chat checks.
The answer should be `4`. Logs must show GPU layers offloaded to the selected backend.

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
curl -f "http://127.0.0.1:$LLAMA_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$LLAMA_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2 + 2? Answer with just the number.\"}],\"temperature\":0,\"max_tokens\":32}"
sudo journalctl -u llama.service -n 100 --no-pager
```

To test the other backend, set `export LLAMA_BACKEND=hip`, then rerun cells
`llama-service` and `llama-verify`. Use `vulkan` to switch back. Both binaries remain
installed; only one model server runs at a time.

Sources: [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/v0.7.0/docs/build.md),
[server options](https://github.com/ggml-org/llama.cpp/blob/v0.7.0/tools/server/README.md),
[Gemma model](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF).
