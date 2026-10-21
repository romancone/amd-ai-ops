# Benchmarks

Latest session: **2026-10-21**. Default model: **ggml-org Gemma 4 12B IT Q4_0**.
RX 6900 XT 16 GiB, Core i9-10900KF, Ubuntu 26.04; llama.cpp `1242be66f`
with the HIP RDNA2 Flash Attention patch. Rates are tokens/s.
FA means `--flash-attn` / `-fa`.

Recommendations: [MODELS.md](MODELS.md). Installation: [llama.md](llama.md).

## 1. Text benchmarks

Matched pp512/tg128 runs: five repetitions, 10 threads, batch 2048, microbatch
512, 99 GPU layers, one sequence and MTP off. The service was stopped during
benchmarks. Values are mean ± standard deviation; all runs passed.

| Backend | K/V cache | FA | pp512 | tg128 |
|---|---|---|---:|---:|
| Vulkan | f16/f16 | on | 1375.17 ± 1.18 | 52.72 ± 0.14 |
| HIP | f16/f16 | on | 1445.21 ± 15.62 | 51.70 ± 0.06 |
| Vulkan | Q8/Q8 | on | 1363.12 ± 1.61 | 50.35 ± 0.23 |
| HIP | Q8/Q8 | on | 1432.80 ± 16.69 | 46.23 ± 0.06 |
| Vulkan | Q4/Q4 | on | 1340.82 ± 2.02 | 50.21 ± 0.43 |
| HIP | Q4/Q4 | on | 1432.83 ± 16.50 | 46.97 ± 0.08 |

f16/f16 was fastest within each backend. With that cache, Vulkan generated tokens
2.0% faster; HIP processed prompts 5.1% faster.

## 2. BF16 image checks

Both backends used f16/f16 cache, FA on, context 4096, one slot, thinking off and
MTP off. The Runme image example sent the same
[newspaper image](media/moon-landing-newspaper.jpg), copied unchanged from
llama.cpp's `tools/mtmd/test-1.jpeg`, with “Describe this image briefly.”,
temperature 0 and a maximum of 256 output tokens. The default microbatch capped
the image budget at 512 tokens.

| Backend | FA | Projector device | Image tokens | Output tokens | Generation t/s | Result |
|---|---|---|---:|---:|---:|---|
| HIP | on | ROCm0 | 130 | 57 | 50.18 | Normal stop; main headline/date correct |
| Vulkan | on | Vulkan0 | 130 | 60 | 48.11 | Normal stop; main headline/date correct |

Both recognized “MEN WALK ON MOON” and July 21, 1969. In the smaller subheadline,
HIP read “PLANET” and Vulkan read “MOON” instead of “PLAIN”. This confirms basic
image inference, not perfect OCR. Different output lengths make these rates
unsuitable for a controlled backend speed comparison.

## 3. Deployment checks

| Check | Result |
|---|---|
| Text API and GPU placement | Both backends passed generation checks and offloaded 49/49 model layers. |
| Projector artifact | Checksum matched; two BF16 tensors and nine FP32 auxiliary tensors, no quantized tensors. |
| Vulkan deployment rerun | `changed=0`, `failed=0` |
| HIP check mode | `changed=0`, `failed=0` |
| Final service | HIP with the BF16 projector, f16/f16 cache and FA on. |

The existing projector file was reused; fresh download was not exercised.
These tests do not establish long-context accuracy, concurrent serving capacity
or general image quality.
