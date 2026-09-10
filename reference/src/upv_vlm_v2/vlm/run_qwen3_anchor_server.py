"""Long-running Qwen3-VL anchor-scoring server for UPV_VLM_v2.

This server is intentionally separate from ``run_qwen_anchor_server.py``.
The existing server is kept for Qwen2.5-VL. This module refuses to load any
model whose ``config.json`` does not declare ``model_type == "qwen3_vl"`` so a
Qwen3 checkpoint cannot be silently instantiated with a Qwen2.5 class.

Run only as a separate service, for example:

    python -m upv_vlm_v2.vlm.run_qwen3_anchor_server \
      --model-path local_models/Qwen3-VL-8B-Instruct \
      --host 127.0.0.1 \
      --port 8898 \
      --load-4bit \
      --device-map auto
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


MODEL: Any | None = None
PROCESSOR: Any | None = None
MODEL_PATH: str | None = None
DEVICE: str | None = None
TRANSFORMERS_VERSION: str | None = None
MAX_MULTI_IMAGES: int = 16


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    """Read a local Hugging Face config.json without importing transformers."""
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found under model_path: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def detect_model_type(model_path: str | Path) -> str:
    """Return the Hugging Face model_type string from config.json."""
    return str(read_model_config(model_path).get("model_type") or "")


def ensure_qwen3_vl_model_path(model_path: str | Path) -> str:
    """Validate that this server is being used only for Qwen3-VL."""
    model_type = detect_model_type(model_path)
    if model_type != "qwen3_vl":
        raise ValueError(
            "run_qwen3_anchor_server only supports model_type=qwen3_vl; "
            f"got model_type={model_type!r} for {model_path}. "
            "Use run_qwen_anchor_server.py for Qwen2.5-VL models."
        )
    return model_type


def transformers_qwen3_support() -> dict[str, Any]:
    """Report whether the installed transformers exposes Qwen3-VL classes."""
    import transformers  # type: ignore

    return {
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "Qwen3VLForConditionalGeneration": hasattr(transformers, "Qwen3VLForConditionalGeneration"),
        "AutoModelForImageTextToText": hasattr(transformers, "AutoModelForImageTextToText"),
        "AutoProcessor": hasattr(transformers, "AutoProcessor"),
    }


def load_model(
    model_path: str,
    load_4bit: bool = False,
    load_8bit: bool = False,
    device_map: str = "auto",
) -> None:
    """Load a Qwen3-VL model once at server startup."""
    global MODEL, PROCESSOR, MODEL_PATH, DEVICE, TRANSFORMERS_VERSION

    ensure_qwen3_vl_model_path(model_path)
    if load_4bit and load_8bit:
        raise ValueError("Use only one of --load-4bit or --load-8bit, not both.")

    print(f"[QWEN3 SERVER] Loading model: {model_path}", flush=True)

    import torch  # type: ignore
    import transformers  # type: ignore
    from transformers import AutoProcessor, BitsAndBytesConfig  # type: ignore

    try:
        from transformers import Qwen3VLForConditionalGeneration as Qwen3Model  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Installed transformers does not expose Qwen3VLForConditionalGeneration. "
            "Do not load Qwen3-VL with the Qwen2.5 server. Upgrade/test in a separate "
            "backup environment before changing the working Qwen2.5 setup."
        ) from exc

    cuda = bool(torch.cuda.is_available())
    dtype = torch.bfloat16 if cuda else torch.float32
    TRANSFORMERS_VERSION = getattr(transformers, "__version__", "unknown")

    print(
        f"[QWEN3 SERVER] transformers={TRANSFORMERS_VERSION} cuda={cuda} dtype={dtype} "
        f"load_4bit={load_4bit} load_8bit={load_8bit} device_map={device_map}",
        flush=True,
    )

    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map if cuda else None,
    }
    if load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda else torch.float32,
        )
    elif load_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = dtype

    MODEL = Qwen3Model.from_pretrained(model_path, **model_kwargs)
    if not cuda:
        MODEL.to("cpu")
    MODEL.eval()
    MODEL_PATH = str(model_path)
    DEVICE = "cuda" if cuda else "cpu"
    print("[QWEN3 SERVER] Model loaded.", flush=True)


def _model_device() -> Any:
    if MODEL is None:
        return "cpu"
    device = getattr(MODEL, "device", None)
    if device is not None:
        return device
    try:
        return next(MODEL.parameters()).device
    except Exception:
        return "cpu"


def _resolve_image_path(request_payload: dict[str, Any]) -> str:
    image_value = request_payload.get("image_path", request_payload.get("image"))
    if not image_value:
        raise KeyError("expected image_path or image in /infer payload")
    if not isinstance(image_value, str):
        raise TypeError("image_path/image must be a local filesystem path string")
    resolved_image = Path(image_value).expanduser().resolve()
    if not resolved_image.exists():
        raise FileNotFoundError(f"image_path not found: {resolved_image}")
    return str(resolved_image)


def _resolve_existing_images(image_paths: Any) -> list[Path]:
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths must be a non-empty list.")
    if len(image_paths) > MAX_MULTI_IMAGES:
        raise ValueError(f"image_paths length {len(image_paths)} exceeds max_multi_images={MAX_MULTI_IMAGES}.")
    resolved: list[Path] = []
    for idx, value in enumerate(image_paths):
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image_paths[{idx}] not found: {path}")
        resolved.append(path)
    return resolved


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    import torch  # type: ignore

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_inference(image_path: str, prompt_text: str, max_new_tokens: int = 900, temperature: float = 0.0) -> str:
    """Run one saved image+prompt inference request."""
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model is not loaded.")

    import torch  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.exists():
        raise FileNotFoundError(f"image_path not found: {resolved_image}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(resolved_image)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    target_device = _model_device()
    inputs = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False if float(temperature) == 0.0 else True,
            temperature=None if float(temperature) == 0.0 else float(temperature),
        )
    input_len = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_len:]
    return PROCESSOR.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def run_multi_image_inference(
    image_paths: Any,
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1400,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[str, list[str]]:
    """Run one ordered multi-image request using the Qwen3-VL chat format."""
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model is not loaded.")

    import torch  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    resolved = _resolve_existing_images(image_paths)
    if not str(user_prompt).strip():
        raise ValueError("user_prompt is required for /infer_multi.")
    messages: list[dict[str, Any]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    content: list[dict[str, str]] = [{"type": "image", "image": str(path)} for path in resolved]
    content.append({"type": "text", "text": str(user_prompt)})
    messages.append({"role": "user", "content": content})

    prompt = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    target_device = _model_device()
    inputs = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    _apply_seed(seed)
    with torch.inference_mode():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=float(temperature) != 0.0,
            temperature=None if float(temperature) == 0.0 else float(temperature),
        )
    input_len = inputs["input_ids"].shape[1]
    text = PROCESSOR.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text, [str(path) for path in resolved]


class Qwen3AnchorRequestHandler(BaseHTTPRequestHandler):
    server_version = "UPV_VLM_v2_Qwen3AnchorServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[QWEN3 SERVER] {self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            return self._send_json(
                {
                    "ok": True,
                    "server": "qwen3_vl_anchor_server",
                    "model_loaded": MODEL is not None and PROCESSOR is not None,
                    "model_path": MODEL_PATH,
                    "device": DEVICE,
                    "transformers_version": TRANSFORMERS_VERSION,
                }
            )
        return self._send_json({"ok": False, "error": "not_found"}, code=404)

    def do_POST(self) -> None:
        endpoint = self.path.rstrip("/")
        if endpoint not in {"/infer", "/infer_multi"}:
            return self._send_json({"ok": False, "error": "expected /infer or /infer_multi"}, code=404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request_payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if endpoint == "/infer_multi":
                seed_value = request_payload.get("seed")
                seed = None if seed_value in {None, ""} else int(seed_value)
                text, resolved = run_multi_image_inference(
                    image_paths=request_payload.get("image_paths"),
                    system_prompt=request_payload.get("system_prompt"),
                    user_prompt=str(request_payload.get("user_prompt") or request_payload.get("prompt_text") or ""),
                    max_new_tokens=int(request_payload.get("max_new_tokens", 1400)),
                    temperature=float(request_payload.get("temperature", 0.0)),
                    seed=seed,
                )
                return self._send_json(
                    {
                        "ok": True,
                        "text": text,
                        "raw_text": text,
                        "num_images": len(resolved),
                        "image_paths": resolved,
                        "generation": {"model_path": MODEL_PATH, "endpoint": "/infer_multi", "seed": seed},
                    }
                )
            image_path = _resolve_image_path(request_payload)
            prompt_text = str(request_payload.get("prompt_text", request_payload.get("prompt", "")))
            if not prompt_text:
                raise KeyError("expected prompt_text or prompt in /infer payload")
            max_new_tokens = int(request_payload.get("max_new_tokens", 900))
            temperature = float(request_payload.get("temperature", 0.0))

            print(f"[QWEN3 SERVER] /infer image={image_path}", flush=True)
            text = run_inference(
                image_path=image_path,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            output_path = request_payload.get("output_path")
            if output_path:
                out = Path(str(output_path))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
            return self._send_json({"ok": True, "text": text, "raw_text": text})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(
                {"ok": False, "error": str(exc), "exception_type": type(exc).__name__},
                code=500,
            )


def create_server(host: str, port: int) -> HTTPServer:
    return HTTPServer((host, int(port)), Qwen3AnchorRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UPV_VLM_v2 Qwen3-VL anchor-selection server.")
    parser.add_argument("--model-path", default="local_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--load-4bit", action="store_true", help="Load model with bitsandbytes 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load model with bitsandbytes 8-bit quantization.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Default: auto.")
    parser.add_argument("--max-multi-images", type=int, default=16)
    parser.add_argument("--check-only", action="store_true", help="Validate model type/import support without loading weights.")
    return parser.parse_args()


def main() -> None:
    global MAX_MULTI_IMAGES
    args = parse_args()
    MAX_MULTI_IMAGES = int(args.max_multi_images)
    ensure_qwen3_vl_model_path(args.model_path)
    support = transformers_qwen3_support()
    print(f"[QWEN3 SERVER] support={json.dumps(support)}", flush=True)
    if not support.get("Qwen3VLForConditionalGeneration"):
        raise RuntimeError(
            "Installed transformers does not support Qwen3VLForConditionalGeneration. "
            "Do not start Qwen3-VL from the Qwen2.5 server."
        )
    if args.check_only:
        print("[QWEN3 SERVER] check-only passed; model weights were not loaded.", flush=True)
        return
    load_model(
        args.model_path,
        load_4bit=bool(args.load_4bit),
        load_8bit=bool(args.load_8bit),
        device_map=str(args.device_map),
    )
    server = create_server(args.host, args.port)
    print(f"[QWEN3 SERVER] Listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
