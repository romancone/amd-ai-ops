# Gemma 4 12B tuning on RX 6900 XT

Vulkan remains the default. The two-user profile uses Q8 keys and values with a
Q4_0 MTP draft; the four-user profile uses Q8 keys and Q4 values without MTP.
Both reserve **262,144 tokens per user**, the model's native limit, using unified
KV storage with an explicit per-slot limit. A **1,152-token microbatch** preserves
the full 1,120-token image budget without the scratch-memory cost of 2,048.

These are the best tested profiles for this workload, not universal optima.
Capacity is allocated and concurrent requests are tested; accuracy across a
completely filled 262,144-token conversation is not established.

## Configuration and method

Measured on 2026-10-20: RX 6900 XT 16 GiB, Core i9-10900KF (10 physical cores),
64 GB system RAM, Ubuntu 26.04, distribution RADV/ROCm packages. Source revision:
[`dd002b91f583af0b80140f9a140dc04856bfc2e0`](https://github.com/ggml-org/llama.cpp/commit/dd002b91f583af0b80140f9a140dc04856bfc2e0),
latest upstream master resolved at the start of the review. All candidates used
the same checksummed ggml-org Gemma 4 12B IT Q4_0 weights and matching sidecars.
Model, draft and cache allocations were checked in GPU-placement logs; VRAM,
GPU activity and power were sampled every 250 ms during timed candidate runs.

The initial screening used simultaneous native `/completion` requests: an
880-token synthetic workshop-inventory prompt and exactly 128 output tokens,
temperature 0, seeds 42 onward, `cache_prompt=false`, `ignore_eos=true`.
Each candidate ran twice after a short arithmetic check. Aggregate throughput is
all generated tokens divided by batch wall time, **including prompt processing**.
It is not the sum of standalone token-generation benchmark rates.

Final OpenAI-compatible API measurements, retrieval checks and visual results
are recorded below. Short screening samples identify useful settings; small
percentage differences should not be treated as statistically established.

## Final chat API results

Three batches per configuration through `/v1/chat/completions`: **895 input tokens
and exactly 128 output tokens per request**, simultaneous requests, temperature 0,
`cache_prompt=false`, `ignore_eos=true`. Both baseline and selected profiles used
the same latest source and model. Baseline used FP16 caches, no MTP, standard
checkpoint settings, microbatch 512 and 131,072 total context tokens. Improvements
below compare the complete profiles, not cache quantization in isolation.

| Users | Cache K / V | MTP | Context per user | Total output tokens/s | Batch completion | Peak VRAM |
|---|---|---|---:|---:|---:|---:|
| 2 baseline | FP16 / FP16 | off | 65,536 | 51.71 | 4.96 s | 10.25 GiB |
| **2 selected** | **Q8 / Q8** | **Q4_0, 3 tokens** | **262,144** | **67.33** | **3.80 s** | **14.60 GiB** |
| 4 baseline | FP16 / FP16 | off | 32,768 | 63.96 | 8.04 s | 11.14 GiB |
| **4 selected** | **Q8 / Q4** | **off** | **262,144** | **70.44** | **7.27 s** | **14.68 GiB** |

Total throughput improved **30.2%** for two users and **10.1%** for four. These are
wall-time results including prefill; they do not establish a speedup for every
prompt length or for standalone decoding. Both selected profiles reached 99% GPU
activity and a sampled peak of 287 W. Normal Vulkan host scratch allocations also
use system RAM; GTT usage alone is not evidence of model-layer CPU offload.

All **six concurrent retrieval checks** returned the correct distinct code after
**11,448 input tokens per request**. Two and four simultaneous image requests also
completed, in 7.43 s and 13.68 s respectively (up to 256 output tokens each).
The full allocated context was not filled with a 262K-token accuracy test.
Ansible deployed both concurrency profiles and the HIP fallback successfully.
The final two-user Vulkan rerun was idempotent: `changed=0`, `failed=0`.
Manual and Ansible service files matched exactly for all three profiles.

### Default-size llama-bench

Latest source, pp512 / tg128, five repetitions, default 10 threads, batch 2,048,
microbatch 512, all layers on GPU. These single-sequence tests do not use MTP.

| Backend | Attention / cache | pp512 tokens/s | tg128 tokens/s |
|---|---|---:|---:|
| Vulkan | FA on, Q8 / Q8 | 1,344.41 ± 1.67 | 50.17 ± 0.16 |
| HIP | FA off, FP16 / FP16 | 1,213.58 ± 7.61 | 49.23 ± 0.05 |

The latest HIP FA-on/Q4-cache trial still aborted with
`GGML_ASSERT(max_blocks_per_sm > 0)`. The role therefore keeps the HIP fallback
at FA off / FP16 with a smaller context. This benchmark does not establish HIP
support for the Vulkan concurrency profiles.

## Visual comparison

Used the existing codex-image-check street-scene image and saved independent
reference, the exact prompt **“Describe this image in detail.”**, temperature 0.7,
8,192 maximum output tokens, and seeds 42/43/44. Only visible answer content was
scored with the [seven-category rubric](image-check.md). All these responses
ended normally; none hit the output limit. The image stays outside the public repo.

| Configuration | Scores at seeds 42 / 43 / 44 | Mean / 100 | Mean response time |
|---|---|---:|---:|
| FP16 control, no MTP, full image budget | 82 / 83 / 73 | 79.3 | 5.77 s |
| Selected two-user profile | 71 / 66 / 75 | 70.7 | 3.31 s |
| Selected four-user profile | 86 / 62 / 65 | 71.0 | 6.04 s |

**Visual functionality passed; equivalent visual quality was not established.**
The selected profiles' mean scores were lower on this fixture. Frequent omissions
were individual warning signs and blue docking stands; some answers misstated
building colors or called the scene a high-angle view. The four-user answers
varied widely between seeds. Scores are subjective, assigned during this review,
and one image with three seeds cannot isolate cache, MTP or sampling effects.
Response times also reflect different answer lengths, so they are not a matched
vision-speed benchmark. Use the runner on representative images before treating
these text-throughput profiles as a vision-quality recommendation.

Numeric measurements and per-category scores are saved in
[the benchmark record](benchmarks/gemma4-rx6900xt-2026-10-20.json).
Private raw responses, logs and temporary measurement harnesses remain in ignored
`results/` on the review workstation.

## Cache and MTP screening

Same 131,072-token total context, Flash Attention on, default 512-token microbatch,
standard checkpoint settings, no MTP:

| KV cache K / V | 2 users, output tokens/s | 4 users, output tokens/s |
|---|---:|---:|
| FP16 / FP16 | 49.09 | 66.29 |
| Q8 / Q8 | 52.68 | 72.06 |
| Q8 / Q4 | 53.44 | 74.06 |
| Q4 / Q4 | 54.15 | 74.70 |

Q4 weights do not require Q4 caches. Q8 retains more cache precision at roughly
half FP16 storage; Q4 values make four full contexts practical. Quantization alone
provided a modest speed improvement here. Q4 keys did not justify their extra
precision loss when Q8 keys fit.

With 262,144 tokens per user, Q8/Q4 caches, a projector loaded, and prompt
checkpoints/RAM prompt caching disabled:

| Draft | Draft length | 2 users, output tokens/s | 4 users, output tokens/s |
|---|---:|---:|---:|
| None | — | 58.68 | 77.97 |
| MTP Q4_0 | 1 | 72.28 | 75.09 |
| MTP Q4_0 | 3 | 72.79 | 69.25 |
| MTP Q8_0 | 3 | 62.17 | 72.04 |

MTP helped two simultaneous requests and regressed four in this workload.
The final two-user profile uses Q8/Q8 caches; draft lengths 2 and 3 were close,
while 4 was slower. Eight CPU threads did not clearly beat the default ten.
Microbatches 256, 512, 1,024, 1,152 and 2,048 were evaluated. Values below 1,120
silently reduce this model's image budget, so the recipe uses 1,152.

## Runtime choices

- `--kv-unified --kv-unified-per-slot 262144`: shares storage while keeping the
  per-user limit. In screening, four-user peak VRAM fell from about 15.65 to
  14.62 GiB at microbatch 1,152. Two-user peak was about 14.60 GiB with MTP.
- `--ctx-checkpoints 0 --cache-ram 0`: removes checkpoint/cached-prompt copying
  for the independent-request workload. Edited or branching chat histories can
  require more prompt reprocessing. This is a workload tradeoff, not a universal
  chat optimization; the upstream defaults are 32 checkpoints and 8,192 MiB.
- `--fit off --n-gpu-layers all`: keeps the measured profile explicit. Reduce
  context on smaller GPUs rather than silently accepting partial model offload.
- `--backend-sampling`: evaluated, not enabled. The server disables it for chat
  requests using the reasoning budget, and native-completion results did not
  consistently improve four-user throughput.
- MTP uses the new `--spec-type draft-mtp`, `--spec-draft-model` and
  `--spec-draft-n-max` options. Older `--draft` aliases have been removed.
  DSpark is not selected: upstream lists Gemma 4 backbone support as planned;
  no matching validated DFlash draft was available in this model bundle.

Use `llama_parallel=2` or `llama_parallel=4` with the role, or the corresponding
Runme profile in [llama.md](llama.md). MTP can be disabled with `llama_mtp=false`
(or `LLAMA_MTP=false` manually) when comparing output quality or workloads where
speculation does not pay off. A stochastic seed does not guarantee identical
outputs across cache types, batching or speculative-decoding configurations.

Sources: [upstream server parameters](https://github.com/ggml-org/llama.cpp/blob/dd002b91f583af0b80140f9a140dc04856bfc2e0/tools/server/README.md),
[speculative decoding](https://github.com/ggml-org/llama.cpp/blob/dd002b91f583af0b80140f9a140dc04856bfc2e0/docs/speculative.md),
[matching Gemma artifacts](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/tree/a29b2a6b584a5bdcb672ccdcda99ea9faeacc51a).
