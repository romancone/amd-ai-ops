# llama.cpp with Vulkan and HIP

A native Gemma 4 12B Q4_0 service with Vulkan by default, a matching image projector,
and MTP for two concurrent users. Build both Vulkan and HIP; run one service.
Thinking is disabled. See [measured profiles](llama-performance.md) and the
[visual check](image-check.md).

Complete [rocm.md](rocm.md) first: generic build/Vulkan libraries and AMD HIP
libraries stay in their separate sections there. This recipe also requires `curl`
(`sudo apt install curl` if it is missing).
Run cells 1.1–1.8 in order as your normal user. Use 1.9 only to change profiles.
Stop on any failed command.
Ansible equivalent: [llama.yml](llama.yml), role [llama](roles/llama/).

## 1. llama.cpp service

### 1.1 Settings

These defaults match [the role](roles/llama/defaults/main.yml). The service binds
to localhost. Defaults target two concurrent users on the 16 GB verification card.
The context is shared equally between users; other cards may need a smaller value.

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
export LLAMA_PARALLEL=2
export LLAMA_CONTEXT=524288
export LLAMA_FLASH_ATTENTION=on
export LLAMA_CACHE_K=q8_0
export LLAMA_CACHE_V=q8_0
export LLAMA_UBATCH=1152
export LLAMA_KV_UNIFIED=true
export LLAMA_CHECKPOINTS=0
export LLAMA_CACHE_RAM=0
export LLAMA_MMPROJ_FILE=mmproj-gemma-4-12B-it-BF16.gguf
export LLAMA_MTP_FILE=mtp-gemma-4-12B-it-Q4_0.gguf
export LLAMA_MTP_TOKENS=3
export LLAMA_VISION=true
export LLAMA_MTP=true
export LLAMA_PORT=8080
```

### 1.2 Get llama.cpp

Pinned upstream revision: `dd002b91f583af0b80140f9a140dc04856bfc2e0` (2026-10-20).
On an existing checkout, fetch and check out this revision instead of cloning again.

<!-- ops-step
id: llama.source.prepare
ansible_task: llama.source.prepare | Prepare the source directory
kind: file
-->
<!-- ops-step
id: llama.source.checkout
ansible_task: llama.source.checkout | Check out the pinned llama.cpp revision
kind: command
-->

```bash { name=llama-source }
sudo mkdir -p "$LLAMA_SOURCE"
sudo chown "$USER" "$LLAMA_SOURCE"
git clone --no-checkout --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SOURCE"
git -C "$LLAMA_SOURCE" fetch --depth 1 origin dd002b91f583af0b80140f9a140dc04856bfc2e0
git -C "$LLAMA_SOURCE" checkout --detach dd002b91f583af0b80140f9a140dc04856bfc2e0
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

The model is 7.22 GB. The BF16 projector enables images; the Q4_0 draft enables
MTP. Keep existing models in their own directories.

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

<!-- ops-step
id: llama.sidecars.download
ansible_task: llama.sidecars.download | Download and verify enabled model sidecars
kind: command
-->

```bash { name=llama-model }
sudo mkdir -p "$LLAMA_MODEL_DIR"
sudo chown "$USER" "$LLAMA_MODEL_DIR"
curl -fL "$LLAMA_MODEL_URL" -o "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
echo "$LLAMA_MODEL_SHA256  $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE" | sha256sum -c -
chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE"
if [ "$LLAMA_VISION" = true ]; then
  curl -fL "${LLAMA_MODEL_URL%/*}/$LLAMA_MMPROJ_FILE" -o "$LLAMA_MODEL_DIR/$LLAMA_MMPROJ_FILE"
  echo "9b1edfa05b634728ca4bfd60b4e6b278e95166c078fa54ae4fa83e680112fd1d  $LLAMA_MODEL_DIR/$LLAMA_MMPROJ_FILE" | sha256sum -c -
  chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MMPROJ_FILE"
fi
if [ "$LLAMA_MTP" = true ]; then
  curl -fL "${LLAMA_MODEL_URL%/*}/$LLAMA_MTP_FILE" -o "$LLAMA_MODEL_DIR/$LLAMA_MTP_FILE"
  echo "b894e614824dfc2746b26d3c3ba78c50000a464382682502392b4325257b7602  $LLAMA_MODEL_DIR/$LLAMA_MTP_FILE" | sha256sum -c -
  chmod 644 "$LLAMA_MODEL_DIR/$LLAMA_MTP_FILE"
fi
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

Flash Attention enables compressed Vulkan caches. A 1,152-token microbatch preserves
the model’s full image budget. Log verbosity 4 records GPU placement.

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
LLAMA_KV_ARGS=""
LLAMA_VISION_ARGS=""
LLAMA_MTP_ARGS=""
if [ "$LLAMA_KV_UNIFIED" = true ]; then
  LLAMA_KV_ARGS=" --kv-unified --kv-unified-per-slot $((LLAMA_CONTEXT / LLAMA_PARALLEL))"
fi
if [ "$LLAMA_VISION" = true ]; then
  LLAMA_VISION_ARGS=" --mmproj $LLAMA_MODEL_DIR/$LLAMA_MMPROJ_FILE"
fi
if [ "$LLAMA_MTP" = true ]; then
  LLAMA_MTP_ARGS=" --spec-type draft-mtp --spec-draft-model $LLAMA_MODEL_DIR/$LLAMA_MTP_FILE --spec-draft-n-max $LLAMA_MTP_TOKENS --spec-draft-ngl all --spec-draft-type-k $LLAMA_CACHE_K --spec-draft-type-v $LLAMA_CACHE_V"
fi
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
ExecStart=$LLAMA_SOURCE/build-$LLAMA_BACKEND/bin/llama-server --model $LLAMA_MODEL_DIR/$LLAMA_MODEL_FILE --alias $LLAMA_MODEL_NAME --ctx-size $LLAMA_CONTEXT --parallel $LLAMA_PARALLEL --flash-attn $LLAMA_FLASH_ATTENTION --cache-type-k $LLAMA_CACHE_K --cache-type-v $LLAMA_CACHE_V --ubatch-size $LLAMA_UBATCH --ctx-checkpoints $LLAMA_CHECKPOINTS --cache-ram $LLAMA_CACHE_RAM --fit off --n-gpu-layers all --host 127.0.0.1 --port $LLAMA_PORT --jinja --reasoning-budget 0 --log-verbosity 4$LLAMA_KV_ARGS$LLAMA_VISION_ARGS$LLAMA_MTP_ARGS
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

### 1.9 Select another profile (Optional)

Starting from `llama-settings`, apply the four-user settings below and rerun `llama-service` and
`llama-verify`. Ansible selects the same profile with `llama_parallel=4`.

```bash { name=llama-four-users }
export LLAMA_PARALLEL=4
export LLAMA_CONTEXT=1048576
export LLAMA_CACHE_V=q4_0
export LLAMA_MTP=false
```

Starting from `llama-settings`, apply the HIP settings below and rerun those same cells. Its
Gemma Flash Attention kernel fails on the verification card, so use FP16 caches.
Ansible selects these backend defaults with `llama_backend=hip`.

```bash { name=llama-hip-profile }
export LLAMA_BACKEND=hip
export LLAMA_KV_UNIFIED=false
export LLAMA_PARALLEL=2
export LLAMA_CONTEXT=4096
export LLAMA_FLASH_ATTENTION=off
export LLAMA_CACHE_K=f16
export LLAMA_CACHE_V=f16
export LLAMA_MTP=false
```

To restore Vulkan defaults, rerun `llama-settings`, `llama-service` and `llama-verify`.
Only one server runs at a time. The profiles disable RAM prompt caching and SWA
checkpoints for the measured independent-request workload; edited or branching
chat histories can require prompt reprocessing. See the performance notes before
changing these settings for a different workload.

Sources: [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/dd002b91f583af0b80140f9a140dc04856bfc2e0/docs/build.md),
[server options](https://github.com/ggml-org/llama.cpp/blob/dd002b91f583af0b80140f9a140dc04856bfc2e0/tools/server/README.md),
[Gemma model](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF).
