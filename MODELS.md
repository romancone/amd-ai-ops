# Models

## Default model

The recommended baseline is **Gemma 4 12B IT Q4_0 with HIP, f16/f16 cache and
Flash Attention on**, a 4,096-token context and one request slot. Thinking and
MTP are off. Image support uses an optional BF16 projector.

Model configuration for [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml):

```yaml
models:  # Models to download
  - name: gemma-4-12B-it-Q4_0
    repo: ggml-org/gemma-4-12B-it-GGUF
    file: gemma-4-12B-it-Q4_0.gguf
    # mmproj: mmproj-gemma-4-12B-it-BF16.gguf  # Optional images; BF16 only

llama_model: gemma-4-12B-it-Q4_0  # Catalog name to serve
llama_backend: hip
llama_flash_attention: "on"
llama_cache_type_k: f16
llama_cache_type_v: f16
llama_context: 4096  # Total context tokens
llama_parallel: 1  # One request slot
```

Deployment and image examples: [llama.md](llama.md).
Measurements and historical comparisons: [BENCHMARK.md](BENCHMARK.md).
