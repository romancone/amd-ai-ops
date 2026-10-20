# Visual check

Compare Gemma's image descriptions using the codex-image-check approach: one
fixed image and prompt, a saved independent reference, and visible answers only.
The llama service must have its matching image projector enabled.

## 1. Describe and review an image

### 1.1 Run the same request

Use your own image and keep it outside this public repository. Run from the
repository root with Python 3; no additional Python packages are required.
For a remote service, forward its localhost port through SSH first.

```bash { name=image-check }
python3 tools/image-check.py --image /path/to/image.jpg --label vulkan-2-users --seed 42
```

The endpoint defaults to `http://127.0.0.1:8080/v1/chat/completions`; override it
with `--endpoint`. The tool discovers a single served model, or accepts `--model`.
It sends **“Describe this image in detail.”**, temperature **0.7**, and a maximum
of **8,192 output tokens**. Use the same seeds when comparing configurations.

Visible text and raw responses are saved separately under ignored `results/`.
The raw file records response time and the actual response model. A successful
HTTP request alone does not establish visual accuracy.

### 1.2 Score against the reference

Inspect the original image and a saved reference from an independent model or
human before scoring. Do not regenerate the reference with the model being tested.
Score only the visible answer; exclude hidden reasoning and timing metadata.

| Category | Points |
|---|---:|
| Overall scene | 15 |
| Foreground | 15 |
| Right side | 15 |
| Left side | 15 |
| Middle distance and background | 15 |
| Lighting and season | 15 |
| Accuracy / absence of invented details | 10 |

Repeat at seeds 42, 43 and 44. Record category scores, omissions, invented details,
response times and configuration together. These scores are subjective checks on
one image, not a general vision benchmark. Keep images and unreviewed responses
out of commits. See [measured configuration results](llama-performance.md).
