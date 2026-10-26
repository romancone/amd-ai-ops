# Model evaluation

Start at [reports/index.html](../reports/index.html). The three report tabs link
to recommendations, model tests and MTP/users; they share `reports/style.css`.
`evaluation/` contains the image questions, frozen reference, scoring rules,
template and report builder.
The GPU runner lives in [roles/llama_evaluation](../roles/llama_evaluation/).

This evaluation adapts the image questions in
[Qengineering/Qwen3.5-9B-NPU](https://github.com/Qengineering/Qwen3.5-9B-NPU#vlm-examples).
That project supplies examples, not a numerical scoring system. The rubric below
is ours; its example answers are not treated as ground truth.

## 1. Automated benchmarks

With both backend builds, model files and BF16 projectors already installed,
`evaluation.yml` repeats the configured GPU matrix and calculates keyword
coverage. It records timing, answers, errors, GPU placement, context and FA status.
Use the MTP mode below for simultaneous users and unified KV.

### 1.1 Matrix

The runner evaluates each entry in `llama_evaluation_models` on both backends
with f16/f16, Q8/Q8 and Q4/Q4 caches. Current measurement settings and model
artifacts are recorded in [BENCHMARK.md](../BENCHMARK.md#shared-test-configuration).
HIP FA occupancy failures receive separate FA-off retries; quantized V-cache
combinations rejected without FA are recorded as unsupported.

Image requests use seed 42, temperature 0.2 and at most 512 output tokens,
without prompt cache reuse. Length-limited output is flagged. Text benchmarks
use `--fit-ctx` with the requested context and `--fit-target 0`; logs must
confirm that allocation and full offload. pp512 does not fill the context.
Set `llama_evaluation_full_context: true` to add an exact token-count capacity
check: fill context minus 128 tokens, then generate 128 without discarding input or
shifting context. The input repeats a short landscape sentence. This measures capacity, not
long-context answer quality or realistic document-processing speed.
VRAM, GTT and process RSS are sampled every second during server checks.

### 1.2 Run

`evaluation.yml` uses the separate `llama_evaluation` role. Example settings for
an ignored inventory override such as `host_vars/gpu-host.yml`:

```yaml
llama_evaluation_dir: /var/tmp/llama-evaluation  # New directory for a new matrix
llama_evaluation_context: 16384  # Override with context on an individual model entry
llama_evaluation_caches: [f16, q8_0, q4_0]
llama_evaluation_full_context: false  # Optional filled-context check
# llama_evaluation_full_context_timeout: 3600  # Allow long 256K fills in MTP mode
# llama_evaluation_mode: mtp-slots-cache  # 16K per user; MTP off/on; 1/2/4 slots
# llama_evaluation_retry_from: /var/tmp/previous-run  # Retry only saved HIP FA failures
llama_evaluation_models:
  - id: gemma12-Q4_0
    family: gemma12
    quant: Q4_0
    file: /opt/models/gemma-4-12B-it-Q4_0/gemma-4-12B-it-Q4_0.gguf
    mmproj: /opt/models/gemma-4-12B-it-Q4_0/mmproj-gemma-4-12B-it-BF16.gguf
    image_min: 70
    image_max: 1120
    ubatch: 1152
    # max_context: 262144  # Required in MTP mode
    # draft: /opt/models/gemma-4-12B-it-Q4_0/mtp-gemma-4-12B-it-Q4_0.gguf
```

Add entries for installed models. Use a new directory when model files, builds
or settings change; reuse a directory only to resume the same evaluation.
Each model can override `context`. Optional `estimated_vram_mib` maps backend
and cache to a memory estimate; values above `llama_evaluation_vram_limit_mib`
are excluded before loading. Filled-context requests time out after 30 minutes
unless `llama_evaluation_full_context_timeout` is changed.

For saved failures, set `llama_evaluation_retry_from` to the old run directory
and `llama_evaluation_dir` to a new one. Only failed HIP FA cases are retried;
successful image checks are not repeated. The original results stay intact.
The runner stops the model service during evaluation and attempts to restart
it afterward if it was active. It does not change service configuration.

```bash
ansible-playbook evaluation.yml --diff --limit gpu-host
```

Raw benchmark JSON, server logs and image responses stay in the run directory.

### 1.3 MTP, users and cache capacity

Set `llama_evaluation_mode: mtp-slots-cache` for the 1/2/4-user matrix.
It fixes 16K **per user**, FA on and unified KV on. It compares f16/f16,
Q8/Q8 and Q4/Q4 caches. With three models, this gives 18 single-user
configurations with MTP off and 72 for 2/4 users with MTP off/on. Each model
needs `max_context`; add `draft` for a separate MTP file. An integrated MTP model
uses the same target file with MTP off and on. Keep projectors BF16.

The runner checks MTP with text and images first, then completes three timing
batches and four image answers per configuration. Capability results are reused.
Servers restart between batches. It records simultaneous slot use, latency,
draft acceptance and memory. Ordinary pp512/tg128 results are saved separately.

With `llama_evaluation_capacity: true`, the default is one f16/f16, MTP-off
capacity check per model/backend, using one slot. To select profiles without
repeating the speed and quality matrix, use:

```yaml
llama_evaluation_mode: mtp-slots-cache
llama_evaluation_capacity_only: true
llama_evaluation_capacity_refine: false  # Faster: validate a safe limit, not the exact maximum
llama_evaluation_capacity_profiles:
  - model: gemma12-Q4_0  # An id in llama_evaluation_models
    backend: hip
    slots: 2
    cache: f16  # Same type for K and V
    mtp: true  # Requires integrated MTP or a configured draft file
llama_evaluation_mtp_guard_tokens: 4
```

Searches start at the model maximum per user unless `start_context` is set.
Refinement searches in 4K steps; safe-limit mode stops at a verified setting.
Every slot must process its full input and generate 128 tokens simultaneously,
without prompt-cache reuse or context shifting. Results distinguish the largest
passing context from a practical setting with 512 MiB GPU spare.

For allocation `C` per user, input is `C - 128` with MTP off. With the four-token
MTP reserve, input is `C - 132` and usable context is `C - 4`.
Total allocation is `C × slots`. An image-memory check also measures workspace,
including in capacity-only mode; it does not score image quality.

The GPU budget is the smaller of detected memory and
`llama_evaluation_vram_limit_mib`. Per-process
[DRM accounting](https://docs.kernel.org/gpu/drm-usage-stats.html#memory) records
requested and resident VRAM. Confirmed eviction stops a failed fill early;
GTT alone does not prove a spill. Device errors remain distinct from memory
limits. Capacity does not measure long-context answer quality.

Saved case results and summaries are reused by path, including failures. Reuse
only when the model hashes, source revision and settings match. Keep failed
attempts intact and use a new case directory for an authorized retry.

Use isolated builds for a new source revision and keep the service build intact.
Quality scoring remains a separate Luna-6 step.

Ansible starts a background job; launch success is not test completion. Inspect
`status-mtp.json` in MTP mode (`status.json` otherwise) and each case's
`result.json`, including service restoration status.
Recorded source revisions identify measured builds, not installation pins.

<a id="image-description-quality"></a>

## 2. Luna-6 quality scoring

[Luna-6 (`gpt-6-luna`)](https://developers.openai.com/api/docs/models/gpt-6-luna) generated
one reference answer for each image using the same prompt, before seeing the
local models' answers. Those [reference answers](reference.json) define **100%**.
This is a comparison with Luna-6 on four tasks, not a claim of perfect visual ability.

Luna-6 reviews saved answers against the images and reference answers, with
model names, backends, caches and earlier scores hidden. Exact duplicate answers
share one judgement. This is a separate Codex step: the GPU runner does not call
Luna-6 or assign description-quality scores.

| Description quality | Points |
|---|---:|
| Factual accuracy and grounded claims | 40 |
| Accurate visible details and relationships | 30 |
| Completion of the image task | 20 |
| Clear, concise wording | 10 |

The configuration score is the mean of four image scores. Failed requests stay
unscored. The report shows answers, score breakdowns and reasons for deductions.
Equivalent descriptions can receive full credit regardless of model size; these
are Luna-6 judgements, not human-verified ground truth.

**Coverage is separate.** It measures keyword matches from [images.json](images.json)
and does not contribute to description quality. It can count a keyword even when
the answer uses it incorrectly. Both quality and coverage have their own sort option.
Scoring saved answers does not rerun image inference.

**Compare in Codex:** select Luna-6 (`gpt-6-luna`); ask it to score anonymized
saved answers against the same images, [frozen reference](reference.json) and
rubric above. No GPU rerun needed.

## 3. Report generation

Copy the run directory to the machine holding this repository. Save Luna-6
judgements as `reviews-luna.json` in that directory, then run:

```bash
python3 evaluation/build_report.py /path/to/run
```

The builder combines saved results and judgements with `reference.json` and
`report.template.html`, then writes `reports/report.html`. It checks that each
judgement matches the saved answer. Missing judgements stay unscored.
Use `--patched-run /path/to/corrected-run` to include corrected results; later
inputs replace earlier rows only for the same model/backend/cache/context/FA.
Use `--additional-run /path/to/native-context-run` to include a new context
matrix. Memory classes distinguish ≤8 GiB, ≤16 GiB and excluded configurations.

This step runs no GPU inference and makes no Luna-6 calls.
`reports/index.html` is fully hand-maintained; the generator never writes it.

For an MTP run, save the blind judgements as `reviews-luna.json` keyed by the
image/prompt/answer hash, then generate its tables and image-answer view:

```bash
python3 evaluation/build_report.py --mtp-run /path/to/run
```

This writes `reports/mtp.html` and sections 5–6 of `BENCHMARK.md`.
The existing single-user results stay intact.

For a capacity-only completion, keep `manifest.json` and `reuse.json` beside
the copied `node2/` results. Pass/fail reports can instead use a verified
`capacity-summary.json` in the run directory. Then run:

```bash
python3 evaluation/build_report.py --capacity-run /path/to/capacity-completion
```

This updates capacity sections while preserving speed tables, image answers
and scores. For context searches, `reuse.json` links completed summaries and the
manifest lists selected profiles; missing results remain pending. Combine
`--mtp-run` and `--capacity-run` to rebuild matched and capacity sections together.
