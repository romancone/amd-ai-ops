# Cached-context timing plan

## 1. Scope and status

Approved on 2026-10-28 for the next session on `codex/mtp-capacity`.
Run on node2; no deployment-default changes, staging, commits or pushes.

Six short checks passed: all three models with MTP off/on, four users,
4,096 cached tokens + 512 new tokens + 128 output tokens. Reuse that validation;
these are not full 16K or 128K measurements. The previous long run was interrupted
at the operator's request and supplies no completed timing result.

## 2. Matrix

**12 cases:** three models × two MTP states × two contexts. HIP, four simultaneous
users, FA on, separate slot caches (`--no-kv-unified`), thinking off.
Keep each model's cache identical across both contexts. Use BF16 projectors only.

| Model | K/V cache | MTP | Context per user | Total allocation |
|---|---|---|---|---|
| Gemma 12B Q4_0 | Q4/Q4 | off, on | 16K, 128K | 65,536 or 524,288 |
| Gemma E4B Q4_0 | f16/f16 | off, on | 16K, 128K | 65,536 or 524,288 |
| Qwen 9B MTP Q4_K_M | Q4/Q4 | off, on | 16K, 128K | 65,536 or 524,288 |

Complete all six 16K cases first. Then run Gemma 12B off/on, E4B off/on,
Qwen off and Qwen on at 128K. Attempt Qwen/MTP-on again; its old unified-KV
allocation failure is not a result for this separate-cache configuration.
No Vulkan expansion, images, quality scoring, baseline pp/tg reruns or capacity searches.

## 3. Workload and acceptance

Use `measure: filled-speed`, `llama_evaluation_capacity_only: true` and explicit
profiles. Set `start_context` to 16,384 or 131,072. Use fresh evidence directories.
The short protocol gate inside each newly loaded server must pass before its fill.

| Allocated per user | MTP | Cached history | New input | Output | Guard |
|---:|---|---:|---:|---:|---:|
| 16,384 | off | 15,744 | 512 | 128 | 0 |
| 16,384 | on | 15,740 | 512 | 128 | 4 |
| 131,072 | off | 130,432 | 512 | 128 | 0 |
| 131,072 | on | 130,428 | 512 | 128 | 4 |

- Verify recorded source, patch, binary and model identities before execution
- Preserve and restore the original service state and unit configuration
- Use real source-code tool output from the saved text fixture, not repeated padding
- Retain cached token IDs; do not re-render the previous assistant turn
- Prepare all requests before the start barrier and submit them concurrently
- Verify exact cache hits, 512 processed input tokens, 128 outputs and overlapping streams for every user
- Keep three draft tokens, full GPU offload, context shifting off and automatic fitting off
- Require GPU-resident buffers and at least 512 MiB GPU headroom
- Record allocation failures as failures; do not substitute cache types or shrink context
- Stop the queue on failure and diagnose before any retry; do not loop over failed cases
- Keep the 90-second short-gate and ten-minute whole-case limits; timeout is not OOM

One measurement per case, no extra repetitions. The twelve-case execution cap is
120 minutes plus service restoration; this is a maximum budget, not an ETA.

## 4. Metrics and reports

Report **TTFT s mean** from request submission to the first token. Report per-user
and aggregate **TG t/s** only over the shared interval from the latest first token
to the earliest last token. Count tokens strictly after the interval starts.
This excludes every user's prefill and first token. Keep cache-fill duration and
end-to-end throughput separate; exclude log writes and monitor shutdown.

Update BENCHMARK.md, reports/index.html and reports/mtp.html, plus affected generator
inputs. Preserve user edits, image scores, capacity results and MTP off/on pairs.
Use only measurements matching the row's model, backend, cache and context.
Do not replace f16 or Vulkan rows with Q4/HIP measurements. Existing 16K `Total t/s`
includes prefill and must not be labelled TG or directly compared as the same metric.

## 5. Evidence and execution references

- Evidence root: `/Users/rshishkin/romancone/amd-ai-ops-evidence-2026-10-25/`
- Passed checks: `cached-short-20261028/`; includes manifest, fixture, results and restored service state
- Node2 copy: `/var/tmp/amd-cached-short-20261028/`
- Frozen source: `/var/tmp/mtp-slots-cache-20261025/source`, revision `d327bf1f0780f87b8c01b367b61853b0c291d7c4`
- Patch diff SHA256: `07feb76af432e44801d2b1e4dd3163ec7850e158d618a60e94b4dd83f457fdc6`
- HIP server SHA256: `a75a931e7a1ba1dc79cd80b6273ab2c675a8ef3e559c4d4a97b6413bc51a8d0c`
- Original service unit SHA256: `31a19110c1d4121b75178647300518edfb577cd547c70b0d9b4f3a78e536bbb4`

Keep raw evidence outside Git. Do not change drivers, packages, build artifacts,
model files or deployment defaults. No reboot.
