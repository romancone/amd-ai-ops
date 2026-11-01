# Models

Model descriptions and configuration examples. GGUF sizes below are weight-file
download sizes in decimal GB; projectors and runtime memory are separate.

Use the YAML examples in [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml),
or the Bash profiles with [llama.md](llama.md). See [BENCHMARK.md](BENCHMARK.md)
for comparisons.

## 1. Gemma 4 12B IT Q4_0

### 1.1 Description

Google DeepMind's instruction-tuned 12B model for conversation, coding and
multimodal understanding. The ggml-org Q4_0 GGUF supports text and image input
with text output. [Model card](https://huggingface.co/google/gemma-4-12B-it).

GGUF weights: **[7.22 GB](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/main/gemma-4-12B-it-Q4_0.gguf)**.

### 1.2 Ansible configuration

Each example starts with one user. For 2 or 4 users, apply all values from the
model's table to the YAML or Bash profile; K/V sets both cache fields.
Total context is 16K × users. With three draft tokens, keep four tokens per user
for MTP: 16,380 usable with MTP on, 16,384 with it off.

| Users (`llama_parallel`) | Backend | K/V cache | MTP (`llama_mtp`) | Total context (`llama_context`) |
|---|---|---|---|---|
| 1 | hip | f16/f16 | true | 16384 |
| 2 | hip | q4_0/q4_0 | true | 32768 |
| 4 | hip | f16/f16 | true | 65536 |

```yaml
models:  # Models to download
  - name: gemma-4-12B-it-Q4_0
    repo: ggml-org/gemma-4-12B-it-GGUF
    file: gemma-4-12B-it-Q4_0.gguf
    draft: mtp-gemma-4-12B-it-Q4_0.gguf  # Matching MTP draft
    mmproj: mmproj-gemma-4-12B-it-BF16.gguf  # BF16 images

llama_model: gemma-4-12B-it-Q4_0  # Catalog name to serve
llama_backend: hip
llama_mtp: true
llama_draft_tokens: 3
llama_kv_unified: true
llama_threads: 10
llama_flash_attention: "on"
llama_cache_type_k: f16
llama_cache_type_v: f16
llama_context: 16384  # Total context tokens
llama_parallel: 1  # One request slot
llama_batch: 2048
llama_ubatch: 1152  # Image microbatch
llama_image_min_tokens: 70
llama_image_max_tokens: 1120
```

### 1.3 Manual / Runme configuration

```bash { name=llama-model-gemma12 }
export LLAMA_MODEL_ID=gemma-4-12B-it-Q4_0
export LLAMA_MODEL_REPO=ggml-org/gemma-4-12B-it-GGUF
export LLAMA_MODEL_FILE=gemma-4-12B-it-Q4_0.gguf
export LLAMA_DRAFT=mtp-gemma-4-12B-it-Q4_0.gguf
export LLAMA_MMPROJ=mmproj-gemma-4-12B-it-BF16.gguf
export LLAMA_CONTEXT=16384
export LLAMA_PARALLEL=1
export LLAMA_MTP=true  # false disables MTP
export LLAMA_DRAFT_TOKENS=3
export LLAMA_KV_UNIFIED=true
export LLAMA_THREADS=10
export LLAMA_BATCH=2048
export LLAMA_UBATCH=1152
export LLAMA_IMAGE_MIN_TOKENS=70
export LLAMA_IMAGE_MAX_TOKENS=1120
export LLAMA_CACHE_TYPE_K=f16
export LLAMA_CACHE_TYPE_V=f16
export LLAMA_FLASH_ATTENTION=on
export LLAMA_BACKEND=hip
export LLAMA_MODEL_DIR=/opt/models/$LLAMA_MODEL_ID
```

## 2. Gemma 4 E4B IT Q4_0 (Optional)

### 2.1 Description

Google DeepMind's instruction-tuned model for local and edge applications.
E4B denotes effective parameter size; embedding tables add further parameters.
The ggml-org Q4_0 GGUF supports text and image input with text output.
[Model card](https://huggingface.co/google/gemma-4-E4B-it).

GGUF weights: **[4.59 GB](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_0.gguf)**.

### 2.2 Ansible configuration

| Users (`llama_parallel`) | Backend | K/V cache | MTP (`llama_mtp`) | Total context (`llama_context`) |
|---|---|---|---|---|
| 1 | vulkan | f16/f16 | true | 16384 |
| 2 | vulkan | f16/f16 | false | 32768 |
| 4 | hip | q4_0/q4_0 | false | 65536 |

Add this entry under `models` alongside Gemma 12B:

```yaml
- name: gemma-4-E4B-it-Q4_0
  repo: ggml-org/gemma-4-E4B-it-GGUF
  file: gemma-4-E4B-it-Q4_0.gguf
  draft: mtp-gemma-4-E4B-it-Q4_0.gguf
  mmproj: mmproj-gemma-4-E4B-it-BF16.gguf
```

Serving configuration:

```yaml
llama_model: gemma-4-E4B-it-Q4_0  # Replaces the existing selection
llama_backend: vulkan
llama_mtp: true
llama_draft_tokens: 3
llama_kv_unified: true
llama_threads: 10
llama_flash_attention: "on"
llama_cache_type_k: f16
llama_cache_type_v: f16
llama_context: 16384
llama_parallel: 1
llama_batch: 2048
llama_ubatch: 1152  # Image microbatch
llama_image_min_tokens: 70
llama_image_max_tokens: 1120
```

### 2.3 Manual / Runme configuration

```bash { name=llama-model-e4b }
export LLAMA_MODEL_ID=gemma-4-E4B-it-Q4_0
export LLAMA_MODEL_REPO=ggml-org/gemma-4-E4B-it-GGUF
export LLAMA_MODEL_FILE=gemma-4-E4B-it-Q4_0.gguf
export LLAMA_DRAFT=mtp-gemma-4-E4B-it-Q4_0.gguf
export LLAMA_MMPROJ=mmproj-gemma-4-E4B-it-BF16.gguf
export LLAMA_CONTEXT=16384
export LLAMA_PARALLEL=1
export LLAMA_MTP=true
export LLAMA_DRAFT_TOKENS=3
export LLAMA_KV_UNIFIED=true
export LLAMA_THREADS=10
export LLAMA_BATCH=2048
export LLAMA_UBATCH=1152
export LLAMA_IMAGE_MIN_TOKENS=70
export LLAMA_IMAGE_MAX_TOKENS=1120
export LLAMA_CACHE_TYPE_K=f16
export LLAMA_CACHE_TYPE_V=f16
export LLAMA_FLASH_ATTENTION=on
export LLAMA_BACKEND=vulkan
export LLAMA_MODEL_DIR=/opt/models/$LLAMA_MODEL_ID
```

## 3. Qwen3.5-9B MTP Q4_K_M (Optional)

### 3.1 Description

Qwen's 9B vision-language model for text generation and image understanding.
Its language model combines Gated DeltaNet and attention layers. This catalog
uses the Unsloth MTP Q4_K_M GGUF with draft weights included.
[Model card](https://huggingface.co/Qwen/Qwen3.5-9B).

GGUF weights: **[5.87 GB](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/blob/main/Qwen3.5-9B-Q4_K_M.gguf)**.

### 3.2 Ansible configuration

| Users (`llama_parallel`) | Backend | K/V cache | MTP (`llama_mtp`) | Total context (`llama_context`) |
|---|---|---|---|---|
| 1 | vulkan | f16/f16 | true | 16384 |
| 2 | hip | f16/f16 | false | 32768 |
| 4 | vulkan | f16/f16 | false | 65536 |

Add this entry under `models`:

```yaml
- name: Qwen3.5-9B-MTP-Q4_K_M
  repo: unsloth/Qwen3.5-9B-MTP-GGUF
  file: Qwen3.5-9B-Q4_K_M.gguf
  mtp_builtin: true  # No separate draft file
  mmproj: mmproj-BF16.gguf  # BF16 projector for images
```

Serving configuration:

```yaml
llama_model: Qwen3.5-9B-MTP-Q4_K_M  # Replaces the existing selection
llama_backend: vulkan
llama_mtp: true
llama_draft_tokens: 3
llama_kv_unified: true
llama_threads: 10
llama_flash_attention: "on"
llama_cache_type_k: f16
llama_cache_type_v: f16
llama_context: 16384
llama_parallel: 1
llama_batch: 2048
llama_ubatch: 2048  # Image processing microbatch
llama_image_min_tokens: 1024
llama_image_max_tokens: 2048
```

### 3.3 Manual / Runme configuration

```bash { name=llama-model-qwen }
export LLAMA_MODEL_ID=Qwen3.5-9B-MTP-Q4_K_M
export LLAMA_MODEL_REPO=unsloth/Qwen3.5-9B-MTP-GGUF
export LLAMA_MODEL_FILE=Qwen3.5-9B-Q4_K_M.gguf
unset LLAMA_DRAFT  # MTP weights are included in the model
export LLAMA_MMPROJ=mmproj-BF16.gguf
export LLAMA_CONTEXT=16384
export LLAMA_PARALLEL=1
export LLAMA_MTP=true
export LLAMA_DRAFT_TOKENS=3
export LLAMA_KV_UNIFIED=true
export LLAMA_THREADS=10
export LLAMA_BATCH=2048
export LLAMA_UBATCH=2048
export LLAMA_IMAGE_MIN_TOKENS=1024
export LLAMA_IMAGE_MAX_TOKENS=2048
export LLAMA_CACHE_TYPE_K=f16
export LLAMA_CACHE_TYPE_V=f16
export LLAMA_FLASH_ATTENTION=on
export LLAMA_BACKEND=vulkan
export LLAMA_MODEL_DIR=/opt/models/$LLAMA_MODEL_ID
```

