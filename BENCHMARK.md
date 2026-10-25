# Benchmarks

Latest tests **2026-10-25** on RX 6900 XT 16 GiB, Core i9-10900KF, Ubuntu 26.04.

[Interactive report](reports/report.html) · [Method](evaluation/README.md) ·
[Model configuration](MODELS.md) · [Runme guide](llama.md)

## Shared test configuration

| Setting | Value |
|---|---|
| Build | llama.cpp `1242be66f`; HIP with RDNA2 and Qwen projector FA fixes |
| Context / request slots | 16,384, or native context in section 2 / 1 |
| Flash Attention | on |
| Thinking / MTP | off / off |
| GPU placement | All model layers and the BF16 projector on GPU |
| Text benchmark | pp512 and tg128, three repetitions |
| Text threads / batch / microbatch | 10 / 2048 / 512 |
| Image checks | Four images, one answer per image and configuration |
| Gemma image microbatch / token budget | 1152 / 70–1120 |
| Qwen image microbatch / token budget | 2048 / 1024–2048 |

Context is the text capacity, measured in tokens (pieces of text).
16K tests reserved this space but used shorter prompts.

## Q8 vs Q4: image quality

**Use Qwen3.5-9B Q4_K_M / Vulkan / f16/f16 / FA on for images.** Q4 / Q8
averaged **97.25 / 96.33 on Vulkan** and **91.83 / 94.17 on HIP**.
Each image ran three times at 16K, f16/f16 and FA on, restarting the server
with an empty cache each time. Gemma E4B Q8 named the wrong movie for Titanic
in all three repeats on both backends.

Tables use one answer per image, separate from these averages.
Scores cover four images only.

## 1. Matched 16K results

Nine model files × two backends × three caches: **54 configurations**.

**pp512:** reading 512 prompt tokens. **tg128:** generating 128 tokens.
Both use tokens/second; higher is faster. Quality averages four image scores
against Luna-6 (`gpt-6-luna`); keyword coverage is separate.
¹ VRAM is GPU memory use at run end. Answers, score breakdowns and speed
variation are in the interactive report.

| Model artifact | Backend | K/V cache | FA | pp512 | tg128 | Image quality | VRAM GiB¹ | Group |
|---|---|---|---|---:|---:|---:|---:|---|
| gemma-e4b-Q4_0 | hip | f16/f16 | on | 3436.03 | 93.10 | 84.25% | 4.99 | ≤8 GiB |
| gemma-e4b-Q4_0 | hip | q8_0/q8_0 | on | 3394.97 | 79.26 | 84.75% | 4.80 | ≤8 GiB |
| gemma-e4b-Q4_0 | hip | q4_0/q4_0 | on | 3399.32 | 80.40 | 86.00% | 4.72 | ≤8 GiB |
| gemma-e4b-Q4_0 | vulkan | f16/f16 | on | 3325.99 | 108.29 | 83.00% | 4.64 | ≤8 GiB |
| gemma-e4b-Q4_0 | vulkan | q8_0/q8_0 | on | 3285.57 | 101.28 | 82.50% | 4.50 | ≤8 GiB |
| gemma-e4b-Q4_0 | vulkan | q4_0/q4_0 | on | 3207.06 | 100.60 | 83.25% | 4.41 | ≤8 GiB |
| gemma-e4b-Q8_0 | hip | f16/f16 | on | 3332.43 | 68.03 | 80.75% | 6.88 | ≤8 GiB |
| gemma-e4b-Q8_0 | hip | q8_0/q8_0 | on | 3288.16 | 60.37 | 80.50% | 6.70 | ≤8 GiB |
| gemma-e4b-Q8_0 | hip | q4_0/q4_0 | on | 3293.96 | 61.10 | 82.50% | 6.62 | ≤8 GiB |
| gemma-e4b-Q8_0 | vulkan | f16/f16 | on | 2600.66 | 77.00 | 81.00% | 6.53 | ≤8 GiB |
| gemma-e4b-Q8_0 | vulkan | q8_0/q8_0 | on | 2569.89 | 73.53 | 80.75% | 6.39 | ≤8 GiB |
| gemma-e4b-Q8_0 | vulkan | q4_0/q4_0 | on | 2530.29 | 73.29 | 82.25% | 6.31 | ≤8 GiB |
| gemma12-Q4_0 | hip | f16/f16 | on | 1442.66 | 51.71 | 98.50% | 8.65 | ≤16 GiB |
| gemma12-Q4_0 | hip | q8_0/q8_0 | on | 1427.95 | 46.24 | 98.50% | 8.21 | ≤16 GiB |
| gemma12-Q4_0 | hip | q4_0/q4_0 | on | 1419.46 | 47.00 | 97.25% | 7.97 | ≤8 GiB |
| gemma12-Q4_0 | vulkan | f16/f16 | on | 1374.62 | 54.91 | 97.25% | 8.38 | ≤16 GiB |
| gemma12-Q4_0 | vulkan | q8_0/q8_0 | on | 1351.48 | 52.85 | 97.25% | 7.93 | ≤8 GiB |
| gemma12-Q4_0 | vulkan | q4_0/q4_0 | on | 1337.53 | 52.68 | 96.75% | 7.69 | ≤8 GiB |
| gemma12-Q8_0 | hip | f16/f16 | on | 1358.50 | 32.66 | 98.50% | 13.72 | ≤16 GiB |
| gemma12-Q8_0 | hip | q8_0/q8_0 | on | 1348.37 | 30.32 | 97.50% | 13.29 | ≤16 GiB |
| gemma12-Q8_0 | hip | q4_0/q4_0 | on | 1337.11 | 30.64 | 97.25% | 13.05 | ≤16 GiB |
| gemma12-Q8_0 | vulkan | f16/f16 | on | 1094.81 | 34.68 | 98.50% | 13.45 | ≤16 GiB |
| gemma12-Q8_0 | vulkan | q8_0/q8_0 | on | 1081.99 | 33.72 | 98.25% | 13.01 | ≤16 GiB |
| gemma12-Q8_0 | vulkan | q4_0/q4_0 | on | 1074.20 | 33.66 | 99.25% | 12.76 | ≤16 GiB |
| gemma12-qat-Q4_0 | hip | f16/f16 | on | 1403.34 | 52.92 | 98.25% | 8.42 | ≤16 GiB |
| gemma12-qat-Q4_0 | hip | q8_0/q8_0 | on | 1388.79 | 47.19 | 98.50% | 7.98 | ≤8 GiB |
| gemma12-qat-Q4_0 | hip | q4_0/q4_0 | on | 1386.60 | 47.99 | 97.25% | 7.74 | ≤8 GiB |
| gemma12-qat-Q4_0 | vulkan | f16/f16 | on | 1362.40 | 56.59 | 97.75% | 8.15 | ≤16 GiB |
| gemma12-qat-Q4_0 | vulkan | q8_0/q8_0 | on | 1342.11 | 54.20 | 98.00% | 7.70 | ≤8 GiB |
| gemma12-qat-Q4_0 | vulkan | q4_0/q4_0 | on | 1329.72 | 53.95 | 98.00% | 7.46 | ≤8 GiB |
| qwen35-Q4_K_M | hip | f16/f16 | on | 1546.50 | 68.18 | 92.75% | 7.33 | ≤8 GiB |
| qwen35-Q4_K_M | hip | q8_0/q8_0 | on | 1520.09 | 66.16 | 92.50% | 7.10 | ≤8 GiB |
| qwen35-Q4_K_M | hip | q4_0/q4_0 | on | 1506.96 | 66.13 | 97.75% | 6.98 | ≤8 GiB |
| qwen35-Q4_K_M | vulkan | f16/f16 | on | 1537.75 | 75.00 | 98.75% | 7.00 | ≤8 GiB |
| qwen35-Q4_K_M | vulkan | q8_0/q8_0 | on | 1534.65 | 73.92 | 96.75% | 6.76 | ≤8 GiB |
| qwen35-Q4_K_M | vulkan | q4_0/q4_0 | on | 1529.78 | 73.80 | 97.75% | 6.63 | ≤8 GiB |
| qwen35-Q8_0 | hip | f16/f16 | on | 2029.59 | 46.41 | 97.00% | 10.43 | ≤16 GiB |
| qwen35-Q8_0 | hip | q8_0/q8_0 | on | 2013.42 | 45.48 | 95.00% | 10.21 | ≤16 GiB |
| qwen35-Q8_0 | hip | q4_0/q4_0 | on | 2020.57 | 45.51 | 96.50% | 10.08 | ≤16 GiB |
| qwen35-Q8_0 | vulkan | f16/f16 | on | 1594.01 | 50.30 | 97.75% | 10.11 | ≤16 GiB |
| qwen35-Q8_0 | vulkan | q8_0/q8_0 | on | 1588.50 | 49.90 | 96.25% | 9.87 | ≤16 GiB |
| qwen35-Q8_0 | vulkan | q4_0/q4_0 | on | 1584.19 | 49.92 | 96.25% | 9.74 | ≤16 GiB |
| qwen35-mtp-Q4_K_M | hip | f16/f16 | on | 1539.75 | 69.37 | 97.50% | 7.35 | ≤8 GiB |
| qwen35-mtp-Q4_K_M | hip | q8_0/q8_0 | on | 1537.01 | 67.31 | 97.75% | 7.12 | ≤8 GiB |
| qwen35-mtp-Q4_K_M | hip | q4_0/q4_0 | on | 1537.45 | 67.30 | 98.25% | 6.99 | ≤8 GiB |
| qwen35-mtp-Q4_K_M | vulkan | f16/f16 | on | 1557.42 | 74.93 | 97.25% | 7.02 | ≤8 GiB |
| qwen35-mtp-Q4_K_M | vulkan | q8_0/q8_0 | on | 1552.14 | 73.95 | 97.25% | 6.78 | ≤8 GiB |
| qwen35-mtp-Q4_K_M | vulkan | q4_0/q4_0 | on | 1548.42 | 73.56 | 97.75% | 6.65 | ≤8 GiB |
| qwen35-mtp-Q8_0 | hip | f16/f16 | on | 2024.75 | 46.25 | 97.00% | 10.43 | ≤16 GiB |
| qwen35-mtp-Q8_0 | hip | q8_0/q8_0 | on | 2011.70 | 45.51 | 95.00% | 10.21 | ≤16 GiB |
| qwen35-mtp-Q8_0 | hip | q4_0/q4_0 | on | 1924.33 | 45.44 | 96.50% | 10.08 | ≤16 GiB |
| qwen35-mtp-Q8_0 | vulkan | f16/f16 | on | 1593.78 | 49.57 | 97.75% | 10.11 | ≤16 GiB |
| qwen35-mtp-Q8_0 | vulkan | q8_0/q8_0 | on | 1587.89 | 48.79 | 96.25% | 9.87 | ≤16 GiB |
| qwen35-mtp-Q8_0 | vulkan | q4_0/q4_0 | on | 1582.74 | 48.92 | 96.25% | 9.74 | ≤16 GiB |

## 2. Q8 at maximum context

**10 passed, 4 failed, 4 excluded.** All 14 eligible pp512/tg128 pairs completed.
52 of 56 image requests succeeded, matching their 16K answers and Luna-6 scores.

Q8 models, one user, FA on. E4B uses 131,072 tokens; Gemma 12B and Qwen use
262,144. Image questions ran before filling the context.

Full pp/tg fills the context with repeated text, leaving room for exactly 128
output tokens. No input is dropped. This checks capacity, not understanding of
long conversations. VRAM is the highest recorded GPU memory use across image
and capacity tests; `~` means estimated.

| Q8 model | Backend | K/V cache | FA | Context | pp512 | tg128 | Image quality % | VRAM GiB | Group | Full pp / tg | Capacity |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| gemma-e4b-Q8_0 | hip | f16/f16 | on | 131,072 | 3384.80 | 68.17 | 80.75 | 8.95 | ≤16 GiB | 1089.36 / 42.63 | pass |
| gemma-e4b-Q8_0 | hip | q8_0/q8_0 | on | 131,072 | 3297.56 | 60.22 | 80.50 | 8.37 | ≤16 GiB | 1070.52 / 24.20 | pass |
| gemma-e4b-Q8_0 | hip | q4_0/q4_0 | on | 131,072 | 3302.20 | 61.09 | 82.50 | 7.85 | ≤8 GiB | 1077.67 / 26.57 | pass |
| gemma-e4b-Q8_0 | vulkan | f16/f16 | on | 131,072 | 2605.14 | 77.68 | 81.00 | 9.44 | ≤16 GiB | 543.98 / 41.61 | pass |
| gemma-e4b-Q8_0 | vulkan | q8_0/q8_0 | on | 131,072 | 2562.43 | 72.12 | 80.75 | 8.09 | ≤16 GiB | 539.10 / 46.33 | pass |
| gemma-e4b-Q8_0 | vulkan | q4_0/q4_0 | on | 131,072 | 2535.04 | 72.05 | 82.25 | 7.08 | ≤8 GiB | 755.62 / 47.50 | pass |
| qwen35-Q8_0 | hip | f16/f16 | on | 262,144 | — | — | — | ~17.93 | >16 GiB | — / — | excluded |
| qwen35-Q8_0 | hip | q8_0/q8_0 | on | 262,144 | 2023.49 | 45.43 | 95.00 | 15.97 | ≤16 GiB | 204.44 / 10.72 | pass |
| qwen35-Q8_0 | hip | q4_0/q4_0 | on | 262,144 | 2019.26 | 45.47 | 96.50 | 14.08 | ≤16 GiB | 214.72 / 12.14 | pass |
| qwen35-Q8_0 | vulkan | f16/f16 | on | 262,144 | — | — | — | ~17.61 | >16 GiB | — / — | excluded |
| qwen35-Q8_0 | vulkan | q8_0/q8_0 | on | 262,144 | 1587.67 | 48.89 | 96.25 | 15.38 | ≤16 GiB | — / — | device lost |
| qwen35-Q8_0 | vulkan | q4_0/q4_0 | on | 262,144 | 1590.96 | 48.80 | 96.25 | 12.80 | ≤16 GiB | 332.94 / 28.28 | pass |
| gemma12-Q8_0 | hip | f16/f16 | on | 262,144 | — | — | — | ~17.47 | >16 GiB | — / — | excluded |
| gemma12-Q8_0 | hip | q8_0/q8_0 | on | 262,144 | 1357.20 | 30.29 | — | 15.97 | not fitting | — / — | image OOM |
| gemma12-Q8_0 | hip | q4_0/q4_0 | on | 262,144 | 1232.40 | 29.78 | 97.25 | 15.08 | ≤16 GiB | 283.75 / 15.03 | pass |
| gemma12-Q8_0 | vulkan | f16/f16 | on | 262,144 | — | — | — | ~17.20 | >16 GiB | — / — | excluded |
| gemma12-Q8_0 | vulkan | q8_0/q8_0 | on | 262,144 | 1089.95 | 33.70 | 98.25 | 15.77 | ≤16 GiB | — / — | device lost |
| gemma12-Q8_0 | vulkan | q4_0/q4_0 | on | 262,144 | 1081.39 | 33.34 | 99.25 | 14.39 | ≤16 GiB | — / — | device lost |

Excluded cases were estimated to exceed 16 GiB. Gemma 12B/HIP with Q8 cache
ran out of memory on images. Vulkan failed while reading prompts at 155K tokens
for Qwen/Q8 cache, 115K for Gemma 12B/Q8 cache and 229K for Gemma 12B/Q4 cache.
Qwen/HIP with Q8 cache passed with only about 12 MiB free.

## 3. What to use

| GPU budget | Image-focused choice | Context | Why |
|---|---|---:|---|
| 8 GiB candidate | Qwen3.5-9B Q4_K_M, Vulkan, f16/f16, FA on | 16K | 98.75% quality, 75.00 t/s; 7.00 GiB end snapshot |
| 8 GiB with more headroom | Same Qwen model, Vulkan, Q4/Q4, FA on | 16K | 97.75%, 73.80 t/s; 6.63 GiB snapshot |
| 16 GiB | Gemma 12B Q4_0, HIP, f16/f16, FA on | 16K | 98.50%, 51.71 t/s; 8.65 GiB snapshot |
| 16 GiB, optional heavier weights | Gemma 12B Q8_0, Vulkan, Q4/Q4, FA on | 16K | Best observed score: 99.25%, but only 33.66 t/s and 12.76 GiB snapshot |
| 8 GiB candidate, Q8 native context | E4B Q8_0, Vulkan, Q4/Q4, FA on | 128K | Passed; 7.08 GiB peak, 47.50 t/s at the limit; image score only 82.25% |
| 16 GiB, Q8 native context | Qwen Q8_0, Vulkan, Q4/Q4, FA on | 256K | Passed; 12.80 GiB peak, 28.28 t/s at the limit |
| 16 GiB, Q8 native context | Gemma 12B Q8_0, HIP, Q4/Q4, FA on | 256K | Passed; 15.08 GiB peak, 15.03 t/s at the limit; Vulkan failed |
| Above 16 GiB | Excluded | Any | Outside this card's memory budget |

**E4B Q8 at 128K:** Vulkan with Q4 cache uses less memory and generates answers
faster; HIP reads the prompt faster. Both still make image-description mistakes.

Model and cache precision can differ. Smaller caches save memory but can affect
answers. MTP and multiple users are not covered by these tests.

## 4. Memory use

16K memory was recorded at run end. Maximum-context tests show the highest
reading taken once per second. Both include other GPU use. All tests used a
16 GiB card; 8 GiB choices need testing on an 8 GiB card. Download size differs
from memory use.

Maximum-context Q8 tests also used this much system RAM:

| Model | Peak process RAM across configurations |
|---|---:|
| E4B | 7.69–7.84 GiB |
| Qwen 9B | 9.05–9.17 GiB |
| Gemma 12B | 12.01–12.16 GiB |

Allow extra RAM for the operating system and other apps. The CSV export includes
process RAM (RSS) and GPU-accessible system memory (GTT); these can overlap,
so do not add them together.
