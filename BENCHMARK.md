# Benchmarks

Tested **2026-10-22** on RX 6900 XT 16 GiB, Core i9-10900KF, Ubuntu 26.04.

[Interactive report](reports/report.html) · [Method](evaluation/README.md) ·
[Model configuration](MODELS.md) · [Runme guide](llama.md)

## Shared test configuration

| Setting | Value |
|---|---|
| Build | llama.cpp `1242be66f`; HIP with RDNA2 and Qwen projector FA fixes |
| Context / request slots | 16,384 tokens / 1 |
| Flash Attention | on |
| Thinking / MTP | off / off |
| GPU placement | All model layers and the BF16 projector on GPU |
| Text benchmark | pp512 and tg128, three repetitions |
| Text threads / batch / microbatch | 10 / 2048 / 512 |
| Image checks | Four images, one answer per image and configuration |
| Gemma image microbatch / token budget | 1152 / 70–1120 |
| Qwen image microbatch / token budget | 2048 / 1024–2048 |

The 16K context was allocated, not filled by a 16K prompt.

## 1. Matched 16K results

Nine model artifacts × two backends × three caches: **54 configurations**.

Rates are tokens/s. Quality is the mean description score against Luna-6 (`gpt-6-luna`)
on four images, not general accuracy. Coverage is separate. Detailed answers,
standard deviations and scoring notes are in the interactive report.

| Model artifact | Backend | K/V cache | FA | pp512 | tg128 | Image quality |
|---|---|---|---|---:|---:|---:|
| gemma-e4b-Q4_0 | hip | f16/f16 | on | 3436.03 | 93.10 | 84.25% |
| gemma-e4b-Q4_0 | hip | q8_0/q8_0 | on | 3394.97 | 79.26 | 84.75% |
| gemma-e4b-Q4_0 | hip | q4_0/q4_0 | on | 3399.32 | 80.40 | 86.00% |
| gemma-e4b-Q4_0 | vulkan | f16/f16 | on | 3325.99 | 108.29 | 83.00% |
| gemma-e4b-Q4_0 | vulkan | q8_0/q8_0 | on | 3285.57 | 101.28 | 82.50% |
| gemma-e4b-Q4_0 | vulkan | q4_0/q4_0 | on | 3207.06 | 100.60 | 83.25% |
| gemma-e4b-Q8_0 | hip | f16/f16 | on | 3332.43 | 68.03 | 80.75% |
| gemma-e4b-Q8_0 | hip | q8_0/q8_0 | on | 3288.16 | 60.37 | 80.50% |
| gemma-e4b-Q8_0 | hip | q4_0/q4_0 | on | 3293.96 | 61.10 | 82.50% |
| gemma-e4b-Q8_0 | vulkan | f16/f16 | on | 2600.66 | 77.00 | 81.00% |
| gemma-e4b-Q8_0 | vulkan | q8_0/q8_0 | on | 2569.89 | 73.53 | 80.75% |
| gemma-e4b-Q8_0 | vulkan | q4_0/q4_0 | on | 2530.29 | 73.29 | 82.25% |
| gemma12-Q4_0 | hip | f16/f16 | on | 1442.66 | 51.71 | 98.50% |
| gemma12-Q4_0 | hip | q8_0/q8_0 | on | 1427.95 | 46.24 | 98.50% |
| gemma12-Q4_0 | hip | q4_0/q4_0 | on | 1419.46 | 47.00 | 97.25% |
| gemma12-Q4_0 | vulkan | f16/f16 | on | 1374.62 | 54.91 | 97.25% |
| gemma12-Q4_0 | vulkan | q8_0/q8_0 | on | 1351.48 | 52.85 | 97.25% |
| gemma12-Q4_0 | vulkan | q4_0/q4_0 | on | 1337.53 | 52.68 | 96.75% |
| gemma12-Q8_0 | hip | f16/f16 | on | 1358.50 | 32.66 | 98.50% |
| gemma12-Q8_0 | hip | q8_0/q8_0 | on | 1348.37 | 30.32 | 97.50% |
| gemma12-Q8_0 | hip | q4_0/q4_0 | on | 1337.11 | 30.64 | 97.25% |
| gemma12-Q8_0 | vulkan | f16/f16 | on | 1094.81 | 34.68 | 98.50% |
| gemma12-Q8_0 | vulkan | q8_0/q8_0 | on | 1081.99 | 33.72 | 98.25% |
| gemma12-Q8_0 | vulkan | q4_0/q4_0 | on | 1074.20 | 33.66 | 99.25% |
| gemma12-qat-Q4_0 | hip | f16/f16 | on | 1403.34 | 52.92 | 98.25% |
| gemma12-qat-Q4_0 | hip | q8_0/q8_0 | on | 1388.79 | 47.19 | 98.50% |
| gemma12-qat-Q4_0 | hip | q4_0/q4_0 | on | 1386.60 | 47.99 | 97.25% |
| gemma12-qat-Q4_0 | vulkan | f16/f16 | on | 1362.40 | 56.59 | 97.75% |
| gemma12-qat-Q4_0 | vulkan | q8_0/q8_0 | on | 1342.11 | 54.20 | 98.00% |
| gemma12-qat-Q4_0 | vulkan | q4_0/q4_0 | on | 1329.72 | 53.95 | 98.00% |
| qwen35-Q4_K_M | hip | f16/f16 | on | 1546.50 | 68.18 | 92.75% |
| qwen35-Q4_K_M | hip | q8_0/q8_0 | on | 1520.09 | 66.16 | 92.50% |
| qwen35-Q4_K_M | hip | q4_0/q4_0 | on | 1506.96 | 66.13 | 97.75% |
| qwen35-Q4_K_M | vulkan | f16/f16 | on | 1537.75 | 75.00 | 98.75% |
| qwen35-Q4_K_M | vulkan | q8_0/q8_0 | on | 1534.65 | 73.92 | 96.75% |
| qwen35-Q4_K_M | vulkan | q4_0/q4_0 | on | 1529.78 | 73.80 | 97.75% |
| qwen35-Q8_0 | hip | f16/f16 | on | 2029.59 | 46.41 | 97.00% |
| qwen35-Q8_0 | hip | q8_0/q8_0 | on | 2013.42 | 45.48 | 95.00% |
| qwen35-Q8_0 | hip | q4_0/q4_0 | on | 2020.57 | 45.51 | 96.50% |
| qwen35-Q8_0 | vulkan | f16/f16 | on | 1594.01 | 50.30 | 97.75% |
| qwen35-Q8_0 | vulkan | q8_0/q8_0 | on | 1588.50 | 49.90 | 96.25% |
| qwen35-Q8_0 | vulkan | q4_0/q4_0 | on | 1584.19 | 49.92 | 96.25% |
| qwen35-mtp-Q4_K_M | hip | f16/f16 | on | 1539.75 | 69.37 | 97.50% |
| qwen35-mtp-Q4_K_M | hip | q8_0/q8_0 | on | 1537.01 | 67.31 | 97.75% |
| qwen35-mtp-Q4_K_M | hip | q4_0/q4_0 | on | 1537.45 | 67.30 | 98.25% |
| qwen35-mtp-Q4_K_M | vulkan | f16/f16 | on | 1557.42 | 74.93 | 97.25% |
| qwen35-mtp-Q4_K_M | vulkan | q8_0/q8_0 | on | 1552.14 | 73.95 | 97.25% |
| qwen35-mtp-Q4_K_M | vulkan | q4_0/q4_0 | on | 1548.42 | 73.56 | 97.75% |
| qwen35-mtp-Q8_0 | hip | f16/f16 | on | 2024.75 | 46.25 | 97.00% |
| qwen35-mtp-Q8_0 | hip | q8_0/q8_0 | on | 2011.70 | 45.51 | 95.00% |
| qwen35-mtp-Q8_0 | hip | q4_0/q4_0 | on | 1924.33 | 45.44 | 96.50% |
| qwen35-mtp-Q8_0 | vulkan | f16/f16 | on | 1593.78 | 49.57 | 97.75% |
| qwen35-mtp-Q8_0 | vulkan | q8_0/q8_0 | on | 1587.89 | 48.79 | 96.25% |
| qwen35-mtp-Q8_0 | vulkan | q4_0/q4_0 | on | 1582.74 | 48.92 | 96.25% |

## 2. Conclusions

- **Gemma 12B Q4_0 / HIP / f16/f16:** 98.50% image quality versus Vulkan's
  97.25%. The difference is one landmark omitted by Vulkan. Average image
  response time was 3.16 s on both; answer lengths differed. Vulkan is faster
  for text generation, while HIP preserves more detail in these saved answers.
- **E4B Q4_0 / HIP / f16/f16:** use HIP for this profile; Vulkan is the faster
  text alternative. Image scores were 84.25% and 83.00%; both missed Titanic.
  Neither backend is recommended when image quality is the priority.
- **Qwen Q4_K_M / Vulkan / f16/f16 for image quality:** 98.75% and 75.00 t/s.
  **Vulkan/Q4/Q4 saves memory:** 97.75% and 73.80 t/s. HIP/Q4 matched its quality
  at 66.13 t/s; HIP/f16 scored 92.75%. Gemma/HIP's 98.50% and Qwen/Vulkan's
  98.75% are too close to establish a meaningful quality winner on four images.

Prefill alone does not justify HIP: for Gemma 12B, it saves 18 ms per 512 input
tokens, while Vulkan saves 145 ms generating 128 tokens. Estimated combined
times from pp512/tg128 are **2.831 / 2.703 s** (HIP/Vulkan) for Gemma 12B/f16,
**1.524 / 1.336 s** for E4B/f16, and **2.275 / 2.069 s** for Qwen/Q4.
These are text timing estimates, not measured image latency.

Four images do not establish long-context accuracy, video quality, concurrent
capacity or a general quality ranking. Q8 and QAT do not show a consistent
reason to replace these Q4-weight profiles. MTP-capable artifacts ran with MTP
off, so this matrix makes no claim about speculative-decoding speedups.

## 3. Memory

End-of-run **16K image** snapshots, including other GPU allocations; these are
not peak process measurements. Budgets are planning estimates, not tested
minimum GPU capacities. Total host RAM requirements remain unmeasured.

| Profile | Occupied VRAM | Planning budget | Logged host buffers |
|---|---:|---:|---:|
| Gemma 12B Q4_0 / HIP / f16 | 8.65 GiB | 10 GiB | 1.10 GiB |
| Gemma 12B Q4_0 / Vulkan / f16 | 8.38 GiB | 10 GiB | 1.10 GiB |
| E4B Q4_0 / Vulkan / f16 | 4.64 GiB | 6 GiB | 2.27 GiB |
| E4B Q4_0 / HIP / f16 | 4.99 GiB | 6 GiB | 2.27 GiB |
| Qwen Q4_K_M / Vulkan / f16 | 7.00 GiB | 8 GiB | 0.72 GiB |
| Qwen Q4_K_M / HIP / f16 | 7.33 GiB | 8 GiB | 0.72 GiB |
| Qwen Q4_K_M / Vulkan / Q4 | 6.63 GiB | 8 GiB | 0.72 GiB |
| Qwen Q4_K_M / HIP / Q4 | 6.98 GiB | 8 GiB | 0.72 GiB |

Host buffers exclude other process allocations and file caching; they are not
total RAM requirements.

## 4. Backend compatibility

Gemma 12B passed deployment checks on **both HIP and Vulkan** with f16/f16
cache, FA on and 16K context. Both served text and loaded the BF16 image
projector on the GPU, with all 49 model layers offloaded.

Deployment instructions: [llama.md](llama.md). These checks confirm service
operation; image quality and speed are compared above.
