# Benchmarks

Latest tests **2026-10-25** on RX 6900 XT 16 GiB, Core i9-10900KF, Ubuntu 26.04.

[Interactive report](reports/report.html) · [Method](evaluation/README.md) ·
[Model configuration](MODELS.md) · [Runme guide](llama.md)

## Shared test configuration

| Setting | Value |
|---|---|
| Build, sections 1–4 | llama.cpp `1242be66f`; HIP with RDNA2 and Qwen projector FA fixes |
| Build, sections 5–6 | llama.cpp `d327bf1f0`; same HIP fixes |
| Context / request slots | Sections 1–4: one user; 16K or native context. Section 5: 16K per user, 1/2/4 users. Section 6: one user, maximum/practical context |
| Flash Attention | on |
| Thinking / MTP | Thinking off. MTP off in sections 1–4 and 6; off/on in section 5 |
| Unified KV | Off in sections 1–4; on in sections 5–6 |
| GPU placement | All target/draft layers and the BF16 projector on GPU |
| Text benchmark | pp512 and tg128, three repetitions |
| llama-bench threads / batch / microbatch | 10 / 2048 / 512 |
| Server timing | 512 input / 128 output tokens per user, three batches; Gemma microbatch 1152, Qwen 2048; 10 threads, batch 2048 |
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
answers. MTP and simultaneous users are covered in section 5.

### MTP and cache choices at 16K per user

Image-first candidates have the highest four-image mean in each group.
The faster alternative shows the speed/quality tradeoff. These are small-sample
choices, not proof that a cache type improves image quality. Single-user MTP-on
candidates use the six preserved checks alongside the 90-configuration matrix.
All use FA and unified KV on, 16K allocated per user and BF16 projectors.

| Model | Users | Image-first candidate | Quality % | Total t/s | GPU GiB | Faster alternative |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 12B Q4 | 1 | Vulkan; Q8/Q8; MTP off | 97.25 | 41.43 | 7.94 | HIP; f16/f16; MTP on: 70.01 t/s, 92.75%, 9.12 GiB |
| Gemma 12B Q4 | 2 | HIP; Q4/Q4; MTP on | 99.25 | 88.82 | 8.71 | HIP; f16/f16; MTP on: 92.34 t/s, 97.25%, 9.77 GiB |
| Gemma 12B Q4 | 4 | HIP; f16/f16; MTP off | 97.50 | 103.73 | 10.53 | HIP; f16/f16; MTP on: 117.42 t/s, 96.75%, 11.05 GiB |
| Gemma E4B Q4 | 1 | HIP; Q8/Q8; MTP off | 79.25 | 63.96 | 4.87 | Vulkan; f16/f16; MTP on: 107.52 t/s, 77.25%, 4.77 GiB |
| Gemma E4B Q4 | 2 | Vulkan; f16/f16; MTP on | 78.00 | 123.88 | 5.11 | HIP; f16/f16; MTP on: 144.57 t/s, 76.75%, 5.53 GiB |
| Gemma E4B Q4 | 4 | HIP; Q4/Q4; MTP on | 79.25 | 164.35 | 5.78 | HIP; f16/f16; MTP off: 194.83 t/s, 74.25%, 5.98 GiB |
| Qwen MTP Q4_K_M | 1 | Vulkan; f16/f16; MTP on | 97.25 | 67.40 | 7.78 | Same profile |
| Qwen MTP Q4_K_M | 2 | HIP; f16/f16; MTP off | 97.75 | 87.49 | 8.07 | Vulkan; f16/f16; MTP off: 93.39 t/s, 93.75%, 7.63 GiB |
| Qwen MTP Q4_K_M | 4 | Vulkan; Q8/Q8; MTP off | 94.75 | 122.50 | 7.93 | Vulkan; f16/f16; MTP off: 132.22 t/s, 94.50%, 8.86 GiB |

f16/f16 gave the highest total throughput in every model/user group. Smaller
caches saved memory but showed no consistent image-quality advantage.
Gemma 12B and Qwen gave stronger image answers than E4B. Qwen MTP helped one
user but slowed output at two and four; Gemma 12B/HIP MTP improved total
throughput by 63%, 29% and 13% at 1/2/4 users with f16/f16. MTP can delay the
first token; compare TTFT in section 5. Capacity checks are separate in section 6.

## 4. Memory use

Sections 1–2 record 16K memory at run end. Section 5 and maximum-context tests show the highest
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

## 5. MTP and simultaneous users

[Full MTP report and image answers](reports/mtp.html). 90/90 configurations recorded; 90 passed.
16K per user means 16K/32K/64K total. Cache: f16/f16, q8_0/q8_0 and q4_0/q4_0; FA and unified KV on;
BF16 projectors; thinking off; at most three draft tokens. Each row has three
simultaneous timing batches (512 input / 128 output tokens per request) and
four image answers once each. Servers restart
between batches. Qwen uses the dedicated MTP-GGUF target in both modes. The timing prompt is a
fixed inventory/checklist task, not a broad workload sample.

Prompt and generation rates are per-user means from server timings. Total t/s includes prompt time;
speedup compares paired total t/s. TTFT means time to first token. Image quality
averages four Luna-6 scores. Small differences do not establish general accuracy.
Each slot allocates 16,384 tokens; the timing/image requests do not fill it.
The four-token MTP guard would leave 16,380 usable tokens per slot, but that
full MTP capacity is not validated here. GPU and process RAM are sampled peaks.
Ordinary pp512/tg128 stays separate.

| Model | Backend | Users | MTP | K/V cache | FA | Prompt t/s/user | Gen t/s/user | Total t/s ± SD | Speedup | TTFT ms | Request s | Draft accept % | Image quality % | Peak GPU GiB | Peak RAM GiB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-e4b-Q4_0 | hip | 1 | off | f16/f16 | on | 1700.67 | 87.43 | 72.91 ± 0.22 | — | 302.23 | 1.75 | — | 74.75 | 5.06 | 3.71 | passed |
| gemma-e4b-Q4_0 | hip | 1 | off | q4_0/q4_0 | on | 1828.70 | 76.13 | 65.62 ± 0.16 | — | 281.75 | 1.95 | — | 73.50 | 4.79 | 4.31 | passed |
| gemma-e4b-Q4_0 | hip | 1 | off | q8_0/q8_0 | on | 1722.05 | 74.72 | 63.96 ± 0.84 | — | 300.80 | 2.00 | — | 79.25 | 4.87 | 4.57 | passed |
| gemma-e4b-Q4_0 | hip | 2 | off | f16/f16 | on | 1131.81 | 79.99 | 125.24 ± 0.63 | — | 454.79 | 2.04 | — | 74.25 | 5.37 | 4.57 | passed |
| gemma-e4b-Q4_0 | hip | 2 | off | q4_0/q4_0 | on | 1089.09 | 68.13 | 109.53 ± 0.46 | — | 471.72 | 2.34 | — | 75.50 | 4.96 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 2 | off | q8_0/q8_0 | on | 1448.94 | 70.69 | 113.19 ± 2.54 | — | 459.30 | 2.26 | — | 75.25 | 5.10 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 2 | on | f16/f16 | on | 947.54 | 106.44 | 144.57 ± 2.79 | 1.15 | 543.25 | 1.74 | 41.79 | 76.75 | 5.53 | 4.48 | passed |
| gemma-e4b-Q4_0 | hip | 2 | on | q4_0/q4_0 | on | 1231.84 | 96.66 | 136.22 ± 6.30 | 1.24 | 529.72 | 1.85 | 46.81 | 77.75 | 5.23 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 2 | on | q8_0/q8_0 | on | 953.16 | 96.01 | 133.22 ± 1.16 | 1.18 | 539.36 | 1.86 | 41.79 | 74.75 | 5.38 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 4 | off | f16/f16 | on | 865.62 | 67.40 | 194.83 ± 1.58 | — | 734.98 | 2.62 | — | 74.25 | 5.98 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 4 | off | q4_0/q4_0 | on | 1033.31 | 60.44 | 179.91 ± 2.50 | — | 726.96 | 2.84 | — | 77.75 | 5.31 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 4 | off | q8_0/q8_0 | on | 836.21 | 61.76 | 180.75 ± 0.90 | — | 766.23 | 2.83 | — | 75 | 5.59 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 4 | on | f16/f16 | on | 1102.56 | 61.18 | 173.35 ± 1.78 | 0.89 | 764.97 | 2.87 | 41.16 | 74.25 | 6.22 | 4.08 | passed |
| gemma-e4b-Q4_0 | hip | 4 | on | q4_0/q4_0 | on | 700.97 | 61.47 | 164.35 ± 1.85 | 0.91 | 904.52 | 2.98 | 43.94 | 79.25 | 5.78 | 4.64 | passed |
| gemma-e4b-Q4_0 | hip | 4 | on | q8_0/q8_0 | on | 880.60 | 59.92 | 165.43 ± 1.29 | 0.92 | 854.75 | 2.99 | 43.84 | 74.75 | 6.06 | 4.64 | passed |
| gemma-e4b-Q4_0 | vulkan | 1 | off | f16/f16 | on | 3049.65 | 96.81 | 86.35 ± 1.00 | — | 169.74 | 1.48 | — | 77.50 | 4.64 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 1 | off | q4_0/q4_0 | on | 3106.93 | 90.86 | 81.76 ± 3.03 | — | 166.36 | 1.57 | — | 73 | 4.41 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 1 | off | q8_0/q8_0 | on | 3092.23 | 91.02 | 81.90 ± 2.42 | — | 166.91 | 1.56 | — | 78.50 | 4.49 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | off | f16/f16 | on | 2164.21 | 84.56 | 142.88 ± 3.20 | — | 283.77 | 1.79 | — | 76.75 | 4.95 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | off | q4_0/q4_0 | on | 2067.42 | 79.22 | 134.15 ± 2.56 | — | 299.43 | 1.90 | — | 74.75 | 4.52 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | off | q8_0/q8_0 | on | 1644.42 | 75.58 | 128.16 ± 7.40 | — | 313.83 | 2.00 | — | 74.50 | 4.67 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | on | f16/f16 | on | 1504.87 | 81.15 | 123.88 ± 2.47 | 0.87 | 406.85 | 1.98 | 38.38 | 78 | 5.11 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | on | q4_0/q4_0 | on | 1310.37 | 70.60 | 114.99 ± 3.11 | 0.86 | 392.72 | 2.19 | 45.03 | 72.50 | 4.69 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 2 | on | q8_0/q8_0 | on | 1565.40 | 70.04 | 111.73 ± 6.10 | 0.87 | 388.79 | 2.21 | 40.12 | 76.75 | 4.84 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | off | f16/f16 | on | 1096.73 | 56.78 | 182.25 ± 12.97 | — | 561.96 | 2.81 | — | 75.25 | 5.56 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | off | q4_0/q4_0 | on | 1516.68 | 49.01 | 163.74 ± 5.20 | — | 514.23 | 3.11 | — | 74.25 | 4.75 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | off | q8_0/q8_0 | on | 1061.60 | 52.02 | 169.13 ± 7.41 | — | 574.03 | 3.02 | — | 75 | 5.03 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | on | f16/f16 | on | 1127.03 | 42.48 | 134.49 ± 1.17 | 0.74 | 629.34 | 3.64 | 42.84 | 77.75 | 5.80 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | on | q4_0/q4_0 | on | 698.49 | 37.48 | 117.77 ± 3.95 | 0.72 | 736.00 | 4.14 | 40.74 | 75 | 4.99 | 4.48 | passed |
| gemma-e4b-Q4_0 | vulkan | 4 | on | q8_0/q8_0 | on | 989.72 | 39.58 | 122.46 ± 8.89 | 0.72 | 711.60 | 3.97 | 43.48 | 77.75 | 5.27 | 4.48 | passed |
| gemma12-Q4_0 | hip | 1 | off | f16/f16 | on | 1309.79 | 49.09 | 42.93 ± 0.41 | — | 394.33 | 2.98 | — | 92.50 | 8.71 | 7.08 | passed |
| gemma12-Q4_0 | hip | 1 | off | q4_0/q4_0 | on | 1368.07 | 44.18 | 39.37 ± 0.20 | — | 376.19 | 3.25 | — | 95.25 | 8.03 | 7.08 | passed |
| gemma12-Q4_0 | hip | 1 | off | q8_0/q8_0 | on | 1313.48 | 42.99 | 38.25 ± 0.04 | — | 391.29 | 3.35 | — | 94.50 | 8.27 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | off | f16/f16 | on | 672.45 | 45.39 | 71.85 ± 0.12 | — | 763.36 | 3.56 | — | 95.75 | 9.33 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | off | q4_0/q4_0 | on | 856.00 | 39.81 | 65.31 ± 0.89 | — | 717.04 | 3.91 | — | 93.75 | 8.24 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | off | q8_0/q8_0 | on | 701.89 | 42.47 | 68.78 ± 0.11 | — | 730.58 | 3.72 | — | 94.50 | 8.62 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | on | f16/f16 | on | 765.45 | 66.02 | 92.34 ± 0.70 | 1.29 | 801.82 | 2.74 | 55.57 | 97.25 | 9.77 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | on | q4_0/q4_0 | on | 764.66 | 62.40 | 88.82 ± 3.79 | 1.36 | 808.71 | 2.86 | 58.11 | 99.25 | 8.71 | 7.08 | passed |
| gemma12-Q4_0 | hip | 2 | on | q8_0/q8_0 | on | 611.01 | 64.50 | 90.41 ± 1.39 | 1.31 | 840.62 | 2.81 | 56.58 | 93.75 | 9.09 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | off | f16/f16 | on | 317.10 | 38.29 | 103.73 ± 0.30 | — | 1616.46 | 4.93 | — | 97.50 | 10.53 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | off | q4_0/q4_0 | on | 514.23 | 33.03 | 97.44 ± 0.63 | — | 1369.96 | 5.24 | — | 96.50 | 8.64 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | off | q8_0/q8_0 | on | 522.42 | 33.18 | 97.03 ± 0.64 | — | 1407.88 | 5.26 | — | 95.25 | 9.30 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | on | f16/f16 | on | 373.46 | 48.55 | 117.42 ± 2.23 | 1.13 | 1642.73 | 4.27 | 57.52 | 96.75 | 11.05 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | on | q4_0/q4_0 | on | 476.18 | 45.30 | 114.10 ± 2.12 | 1.17 | 1510.72 | 4.39 | 58.70 | 93 | 9.25 | 7.08 | passed |
| gemma12-Q4_0 | hip | 4 | on | q8_0/q8_0 | on | 466.28 | 45.33 | 113.59 ± 1.70 | 1.17 | 1543.57 | 4.39 | 59.21 | 95.25 | 9.91 | 7.08 | passed |
| gemma12-Q4_0 | vulkan | 1 | off | f16/f16 | on | 1333.35 | 50.69 | 44.24 ± 0.38 | — | 385.78 | 2.89 | — | 94.50 | 8.38 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 1 | off | q4_0/q4_0 | on | 1319.62 | 48.41 | 42.47 ± 1.74 | — | 389.52 | 3.02 | — | 93.75 | 7.69 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 1 | off | q8_0/q8_0 | on | 1329.12 | 47.05 | 41.43 ± 0.21 | — | 387.13 | 3.09 | — | 97.25 | 7.94 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | off | f16/f16 | on | 699.88 | 45.52 | 72.61 ± 0.58 | — | 733.47 | 3.52 | — | 95.25 | 9.00 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | off | q4_0/q4_0 | on | 681.63 | 43.39 | 69.49 ± 0.57 | — | 753.17 | 3.68 | — | 94 | 7.91 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | off | q8_0/q8_0 | on | 686.83 | 43.45 | 69.69 ± 0.47 | — | 748.00 | 3.67 | — | 93.50 | 8.30 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | on | f16/f16 | on | 624.24 | 55.73 | 81.97 ± 1.06 | 1.13 | 822.73 | 3.10 | 58.12 | 96.50 | 9.44 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | on | q4_0/q4_0 | on | 826.10 | 48.29 | 75.20 ± 1.26 | 1.08 | 743.46 | 3.38 | 56.52 | 94.50 | 8.36 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 2 | on | q8_0/q8_0 | on | 655.44 | 51.75 | 78.17 ± 2.00 | 1.12 | 782.57 | 3.24 | 60.45 | 94 | 8.74 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | off | f16/f16 | on | 548.51 | 30.49 | 92.57 ± 5.91 | — | 1321.70 | 5.53 | — | 93.25 | 10.21 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | off | q4_0/q4_0 | on | 557.33 | 27.65 | 86.89 ± 0.84 | — | 1264.23 | 5.88 | — | 92.50 | 8.31 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | off | q8_0/q8_0 | on | 650.46 | 27.61 | 87.16 ± 1.52 | — | 1224.99 | 5.86 | — | 93.50 | 8.98 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | on | f16/f16 | on | 573.91 | 26.67 | 82.04 ± 0.76 | 0.89 | 1382.90 | 6.18 | 57.44 | 94 | 10.73 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | on | q4_0/q4_0 | on | 503.92 | 27.30 | 81.66 ± 1.31 | 0.94 | 1424.70 | 6.12 | 60.01 | 93.50 | 8.83 | 6.93 | passed |
| gemma12-Q4_0 | vulkan | 4 | on | q8_0/q8_0 | on | 597.52 | 25.92 | 80.97 ± 1.01 | 0.93 | 1295.88 | 6.22 | 56.47 | 93 | 9.50 | 6.93 | passed |
| qwen35-MTP-Q4_K_M | hip | 1 | off | f16/f16 | on | 1046.63 | 67.09 | 53.67 ± 0.41 | — | 491.00 | 2.38 | — | 91.25 | 7.46 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 1 | off | q4_0/q4_0 | on | 1100.61 | 64.72 | 52.67 ± 0.26 | — | 466.75 | 2.43 | — | 93.50 | 7.10 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 1 | off | q8_0/q8_0 | on | 1056.21 | 64.55 | 52.14 ± 0.55 | — | 487.19 | 2.45 | — | 95.50 | 7.22 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | off | f16/f16 | on | 845.88 | 58.73 | 87.49 ± 1.04 | — | 752.55 | 2.92 | — | 97.75 | 8.07 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | off | q4_0/q4_0 | on | 667.33 | 58.40 | 86.93 ± 0.29 | — | 768.90 | 2.94 | — | 91.75 | 7.35 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | off | q8_0/q8_0 | on | 661.51 | 59.14 | 87.52 ± 0.43 | — | 776.37 | 2.92 | — | 95.25 | 7.60 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | on | f16/f16 | on | 816.49 | 54.15 | 76.25 ± 2.98 | 0.87 | 948.74 | 3.31 | 52.95 | 94.50 | 9.02 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | on | q4_0/q4_0 | on | 681.50 | 57.21 | 78.03 ± 2.73 | 0.90 | 952.88 | 3.20 | 55.32 | 93 | 8.30 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 2 | on | q8_0/q8_0 | on | 683.64 | 54.53 | 76.64 ± 2.96 | 0.88 | 945.37 | 3.30 | 52.67 | 91.75 | 8.58 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | off | f16/f16 | on | 358.51 | 49.27 | 127.61 ± 0.42 | — | 1431.27 | 4.01 | — | 94.25 | 9.29 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | off | q4_0/q4_0 | on | 569.88 | 45.98 | 125.45 ± 0.76 | — | 1284.67 | 4.07 | — | 87.75 | 7.95 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | off | q8_0/q8_0 | on | 478.59 | 46.82 | 126.28 ± 0.09 | — | 1328.18 | 4.05 | — | 90.50 | 8.45 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | on | f16/f16 | on | 555.36 | 36.95 | 100.18 ± 1.40 | 0.79 | 1507.06 | 4.99 | 51.80 | 87.25 | 10.73 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | on | q4_0/q4_0 | on | 372.30 | 39.32 | 100.57 ± 0.70 | 0.80 | 1679.92 | 4.94 | 52.29 | 90.75 | 9.50 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | hip | 4 | on | q8_0/q8_0 | on | 556.50 | 36.82 | 99.26 ± 0.54 | 0.79 | 1499.87 | 4.99 | 52.55 | 80 | 10.06 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 1 | off | f16/f16 | on | 1517.51 | 70.19 | 59.53 ± 0.45 | — | 339.32 | 2.15 | — | 94 | 7.01 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 1 | off | q4_0/q4_0 | on | 1509.28 | 67.16 | 57.30 ± 0.32 | — | 341.08 | 2.23 | — | 91 | 6.65 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 1 | off | q8_0/q8_0 | on | 1512.96 | 67.65 | 57.68 ± 0.80 | — | 340.30 | 2.22 | — | 93.75 | 6.78 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | off | f16/f16 | on | 1019.06 | 59.85 | 93.39 ± 2.09 | — | 605.67 | 2.74 | — | 93.75 | 7.63 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | off | q4_0/q4_0 | on | 1019.55 | 57.02 | 90.02 ± 1.33 | — | 605.05 | 2.84 | — | 91.75 | 6.90 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | off | q8_0/q8_0 | on | 1023.11 | 58.24 | 91.64 ± 0.88 | — | 602.00 | 2.79 | — | 93 | 7.16 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | on | f16/f16 | on | 589.97 | 61.82 | 86.63 ± 7.76 | 0.93 | 869.81 | 2.94 | 49.18 | 94.50 | 8.58 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | on | q4_0/q4_0 | on | 937.16 | 59.12 | 86.32 ± 2.22 | 0.96 | 758.67 | 2.93 | 51.91 | 93 | 7.75 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 2 | on | q8_0/q8_0 | on | 620.97 | 64.51 | 90.40 ± 1.50 | 0.99 | 826.34 | 2.80 | 55.83 | 94 | 8.05 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | off | f16/f16 | on | 514.53 | 48.18 | 132.22 ± 2.77 | — | 1218.83 | 3.87 | — | 94.50 | 8.86 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | off | q4_0/q4_0 | on | 767.58 | 41.40 | 123.94 ± 4.93 | — | 1016.63 | 4.12 | — | 84.25 | 7.41 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | off | q8_0/q8_0 | on | 773.79 | 40.50 | 122.50 ± 1.12 | — | 1004.12 | 4.17 | — | 94.75 | 7.93 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | on | f16/f16 | on | 491.27 | 27.02 | 81.65 ± 4.18 | 0.62 | 1463.91 | 6.21 | 51.63 | 93 | 10.27 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | on | q4_0/q4_0 | on | 487.52 | 25.46 | 76.98 ± 0.60 | 0.62 | 1445.31 | 6.47 | 51.28 | 87 | 8.64 | 5.64 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 4 | on | q8_0/q8_0 | on | 319.82 | 25.91 | 76.92 ± 1.34 | 0.63 | 1602.60 | 6.52 | 48.70 | 92.50 | 9.24 | 5.64 | passed |

### Earlier single-user MTP-on results

Saved f16/f16 checks, outside the approved 90-configuration matrix.

| Model | Backend | Users | MTP | K/V cache | FA | Prompt t/s/user | Gen t/s/user | Total t/s ± SD | Speedup | TTFT ms | Request s | Draft accept % | Image quality % | Peak GPU GiB | Peak RAM GiB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-e4b-Q4_0 | hip | 1 | on | f16/f16 | on | 1330.51 | 138.53 | 98.12 ± 3.99 | 1.35 | 388.32 | 1.31 | 39.66 | 74.75 | 5.19 | 3.70 | passed |
| gemma-e4b-Q4_0 | vulkan | 1 | on | f16/f16 | on | 1819.25 | 140.46 | 107.52 ± 1.43 | 1.25 | 285.32 | 1.19 | 36.46 | 77.25 | 4.77 | 4.48 | passed |
| gemma12-Q4_0 | hip | 1 | on | f16/f16 | on | 1094.98 | 93.65 | 70.01 ± 2.46 | 1.63 | 472.56 | 1.83 | 54.17 | 92.75 | 9.12 | 7.09 | passed |
| gemma12-Q4_0 | vulkan | 1 | on | f16/f16 | on | 1071.32 | 88.59 | 66.74 ± 2.34 | 1.51 | 483.21 | 1.92 | 54.86 | 93 | 8.78 | 6.93 | passed |
| qwen35-MTP-Q4_K_M | hip | 1 | on | f16/f16 | on | 756.60 | 97.47 | 64.60 ± 0.74 | 1.20 | 677.83 | 1.98 | 59.12 | 91.25 | 8.23 | 5.77 | passed |
| qwen35-MTP-Q4_K_M | vulkan | 1 | on | f16/f16 | on | 922.89 | 94.76 | 67.40 ± 0.62 | 1.13 | 557.45 | 1.90 | 53.79 | 97.25 | 7.78 | 5.64 | passed |

### Ordinary llama-bench at 16K

Three pp512/tg128 repetitions, MTP off, one user, microbatch 512.
These are separate from server MTP timings.

| Model / backend | FA | Cache | pp512 ± SD | tg128 ± SD |
| --- | --- | --- | --- | --- |
| gemma-e4b-Q4_0-hip | on | f16/f16 | 3466.27 ± 24.02 | 93.23 ± 0.05 |
| gemma-e4b-Q4_0-vulkan | on | f16/f16 | 3298.98 ± 27.96 | 107.94 ± 0.70 |
| gemma12-Q4_0-hip | on | f16/f16 | 1419.88 ± 13.09 | 51.64 ± 0.03 |
| gemma12-Q4_0-vulkan | on | f16/f16 | 1367.14 ± 1.51 | 55.12 ± 0.09 |
| qwen35-MTP-Q4_K_M-hip | on | f16/f16 | 1546.23 ± 2.81 | 69.42 ± 0.05 |
| qwen35-MTP-Q4_K_M-vulkan | on | f16/f16 | 1552.84 ± 1.46 | 74.80 ± 0.07 |

## 6. Cache capacity

6/6 checks recorded: 5 validated, 1 unresolved. One user, MTP off, f16/f16. Status: finished. Context values are exact tokens.
Allocated is the KV pool per user; usable is input plus output.
These six checks use MTP off, so allocated and usable context are equal.
Input = allocated minus 128; output = 128. No MTP capacity claim is made.
Memory refers to the largest passing context, or the failed native attempt when no pass exists. Practical GPU free is the
smallest sampled headroom during the practical full fill and image check;
it must be at least 512 MiB. Memory is sampled once per second.
Lower bounds are not proven maxima. Upper bounds may be tested failures or
conservative memory exclusions. Capacity does not establish long-context image quality.
KV MiB sums logged GPU KV buffers; recurrent state is separate.
GTT includes host staging buffers and is not, by itself, proof of GPU memory spill.
RAM and GPU-accessible host memory (GTT) can overlap; do not add them together.

Earlier unguarded MTP boundary checks (outside the six checks): gemma-e4b-Q4_0/hip, 1 user, f16, 131,072 tokens: generated 127 of 128 requested; gemma-e4b-Q4_0/hip, 1 user, f16, 126,976 tokens: generated 126 of 128 requested. These are not memory failures or capacity passes.

qwen35-MTP-Q4_K_M/vulkan: ErrorDeviceLost at 258,048, 262,144 allocated tokens. No full-capacity limit is established; its 16K timing and image tests passed.

| Model | Backend | FA | Max allocated / usable tokens | KV MiB | Practical allocated / usable tokens | Practical GPU free MiB | Upper test tokens | Full-prompt t/s | Generation t/s | Peak GPU GiB | Peak RAM GiB | Peak GTT GiB | Boundary / status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-e4b-Q4_0 | hip | on | 131,072 / 131,072 | 2118.00 | 131,072 / 131,072 | 9186.73 | 131072 | 1088.30 | 51.56 | 7.01 | 4.57 | 0.01 | model limit |
| gemma-e4b-Q4_0 | vulkan | on | 131,072 / 131,072 | 2118.00 | 131,072 / 131,072 | 9093.97 | 131072 | 564.19 | 49.07 | 7.10 | 4.48 | 0.39 | model limit |
| gemma12-Q4_0 | hip | on | 262,144 / 262,144 | 4816.00 | 262,144 / 262,144 | 3088.29 | 262144 | 282.41 | 24.14 | 12.97 | 7.08 | 0.01 | model limit |
| gemma12-Q4_0 | vulkan | on | 262,144 / 262,144 | 4816.00 | 262,144 / 262,144 | 3372.27 | 262144 | 146.58 | 31.18 | 12.69 | 6.93 | 0.65 | model limit |
| qwen35-MTP-Q4_K_M | hip | on | 262,144 / 262,144 | 8192.00 | 249,856 / 249,856 | 525.65 | 262144 | 205.88 | 29.21 | 15.89 | 5.77 | 0.01 | model limit |
| qwen35-MTP-Q4_K_M | vulkan | on | — | 8192.00 | — | — | 262144 | — | — | 15.67 | 5.64 | 1.40 | not established: device_lost |
