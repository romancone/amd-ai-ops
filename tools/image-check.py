#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_PROMPT = "Describe this image in detail."
MAX_TOKENS = 8192
TEMPERATURE = 0.7
RESPONSES_DIR = Path("./results/image-check/responses")
RAW_RESPONSES_DIR = Path("./results/image-check/raw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one image-description request against an OpenAI-compatible endpoint."
    )
    parser.add_argument("--image", required=True, help="Local image path; never committed by this tool.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for matched comparisons.")
    parser.add_argument("--label", default="visual", help="Configuration label for comparing runs.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Chat completions endpoint.")
    parser.add_argument(
        "--model",
        help="Request model name. If omitted, the script tries GET /v1/models.",
    )
    return parser.parse_args()


def auth_header():
    api_key = os.environ.get("OPENAI_API_KEY", "no-key")
    return {"Authorization": f"Bearer {api_key}"}


def models_endpoint(chat_endpoint):
    parsed = urllib.parse.urlsplit(chat_endpoint)
    path = parsed.path
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")] + "/models"
    else:
        path = path.rstrip("/") + "/models"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def get_json(url):
    request = urllib.request.Request(url, headers=auth_header(), method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", **auth_header()}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_model(endpoint):
    data = get_json(models_endpoint(endpoint)).get("data", [])
    model_ids = [item.get("id") for item in data if item.get("id")]

    if len(model_ids) == 1:
        return model_ids[0]

    if not model_ids:
        raise RuntimeError("GET /v1/models returned no model ids. Pass --model manually.")

    available = "\n".join(f"  - {model_id}" for model_id in model_ids)
    raise RuntimeError(
        "GET /v1/models returned multiple models. Pass --model manually.\n"
        f"Available models:\n{available}"
    )


def encode_image(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
    }


def build_payload(model, image_path, seed):
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": seed,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_PROMPT},
                    encode_image(image_path),
                ],
            }
        ],
    }


def visible_output(response):
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "value"


def save_outputs(image_path, model, response, label, elapsed_s, seed):
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    response_model = response.get("model") or model
    basename = f"{stamp}_{safe_name(label)}_{safe_name(image_path.stem)}_{safe_name(response_model)}"

    json_path = RAW_RESPONSES_DIR / f"{basename}.json"
    text_path = RESPONSES_DIR / f"{basename}.output.txt"

    json_path.write_text(json.dumps({"response": response, "elapsed_s": elapsed_s, "label": label,
                                     "prompt": DEFAULT_PROMPT, "temperature": TEMPERATURE,
                                     "max_tokens": MAX_TOKENS, "seed": seed}, indent=2), encoding="utf-8")
    text_path.write_text(visible_output(response), encoding="utf-8")

    return json_path, text_path


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Missing image: {image_path}", file=sys.stderr)
        return 2

    try:
        model = args.model or discover_model(args.endpoint)
        start = time.perf_counter()
        response = post_json(args.endpoint, build_payload(model, image_path, args.seed))
        elapsed_s = time.perf_counter() - start
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    json_path, text_path = save_outputs(image_path, model, response, f"{args.label}-seed{args.seed}", elapsed_s, args.seed)

    print(f"Image: {image_path}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Request model: {model}")
    print(f"Response model: {response.get('model', '')}")
    print(f"Response time: {elapsed_s:.1f} s")
    print(f"Raw JSON: {json_path}")
    print(f"Raw output: {text_path}")
    print()
    print(visible_output(response))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
