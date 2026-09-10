"""Long-running Qwen anchor-selection server for UPV_VLM_v2.

This module intentionally does not load model dependencies at import time.
Run it as a separate service:

    python -m upv_vlm_v2.vlm.run_qwen_anchor_server \
      --model-path local_models/Qwen2.5-VL-3B-Instruct \
      --host 127.0.0.1 \
      --port 8899
"""

from __future__ import annotations

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


MODEL: Any | None = None
PROCESSOR: Any | None = None
MODEL_PATH: str | None = None
DEVICE: str | None = None
MAX_MULTI_IMAGES: int | None = None


def load_model(
    model_path: str,
    load_4bit: bool = False,
    load_8bit: bool = False,
    device_map: str = "auto",
) -> None:
    """Load Qwen once at server startup.

    For Qwen2.5-VL-32B on a 24 GB RTX 4090, use:
        --load-4bit --device-map auto
    """
    global MODEL, PROCESSOR, MODEL_PATH, DEVICE

    print(f"[QWEN SERVER] Loading model: {model_path}", flush=True)

    import torch  # type: ignore
    from transformers import AutoProcessor, BitsAndBytesConfig  # type: ignore

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel  # type: ignore
    except Exception:
        from transformers import AutoModelForVision2Seq as QwenModel  # type: ignore

    if load_4bit and load_8bit:
        raise ValueError("Use only one of --load-4bit or --load-8bit, not both.")

    cuda = bool(torch.cuda.is_available())
    dtype = torch.bfloat16 if cuda else torch.float32

    print(
        f"[QWEN SERVER] cuda={cuda} dtype={dtype} "
        f"load_4bit={load_4bit} load_8bit={load_8bit} device_map={device_map}",
        flush=True,
    )

    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_kwargs = {
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

    MODEL = QwenModel.from_pretrained(model_path, **model_kwargs)

    if not cuda:
        MODEL.to("cpu")

    MODEL.eval()
    MODEL_PATH = str(model_path)
    DEVICE = "cuda" if cuda else "cpu"
    print("[QWEN SERVER] Model loaded.", flush=True)


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


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    try:
        import numpy as np  # type: ignore

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def run_inference(
    image_path: str,
    prompt_text: str,
    max_new_tokens: int = 900,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    """Run one image+prompt inference request."""
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Qwen model is not loaded.")

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
    _apply_seed(seed)

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


def _resolve_existing_images(image_paths: list[Any]) -> list[Path]:
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths must be a non-empty list.")
    if MAX_MULTI_IMAGES is not None and len(image_paths) > int(MAX_MULTI_IMAGES):
        raise ValueError(f"image_paths length {len(image_paths)} exceeds max_multi_images={MAX_MULTI_IMAGES}.")

    resolved_images = []
    for idx, image_path in enumerate(image_paths, start=1):
        resolved = Path(str(image_path)).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"image_paths[{idx - 1}] not found: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"image_paths[{idx - 1}] is not a file: {resolved}")
        resolved_images.append(resolved)
    return resolved_images


def run_multi_image_inference(
    image_paths: list[Any],
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1400,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[str, list[str]]:
    """Run one prompt against a variable-length ordered list of images."""
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Qwen model is not loaded.")

    import torch  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    resolved_images = _resolve_existing_images(image_paths)
    if not str(user_prompt).strip():
        raise ValueError("user_prompt or prompt_text is required for /infer_multi.")

    messages: list[dict[str, Any]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})

    user_content: list[dict[str, str]] = [
        {"type": "image", "image": str(resolved_image)} for resolved_image in resolved_images
    ]
    user_content.append({"type": "text", "text": str(user_prompt)})
    messages.append({"role": "user", "content": user_content})

    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if image_inputs is None or len(image_inputs) != len(resolved_images):
        raise RuntimeError(
            f"Expected {len(resolved_images)} processed image inputs, got "
            f"{0 if image_inputs is None else len(image_inputs)}."
        )
    inputs = PROCESSOR(
        text=[text],
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
            do_sample=False if float(temperature) == 0.0 else True,
            temperature=None if float(temperature) == 0.0 else float(temperature),
        )
    input_len = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_len:]
    decoded = PROCESSOR.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded, [str(path) for path in resolved_images]


class QwenAnchorRequestHandler(BaseHTTPRequestHandler):
    server_version = "UPV_VLM_v2_QwenAnchorServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[QWEN SERVER] {self.address_string()} - {fmt % args}", flush=True)

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
                    "model_loaded": MODEL is not None and PROCESSOR is not None,
                    "model_path": MODEL_PATH,
                    "device": DEVICE,
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
                image_paths = request_payload.get("image_paths")
                user_prompt = request_payload.get("user_prompt")
                if user_prompt is None:
                    user_prompt = request_payload.get("prompt_text")
                if user_prompt is None:
                    raise ValueError("user_prompt is required, or provide prompt_text as a fallback.")
                system_prompt = request_payload.get("system_prompt")
                max_new_tokens = int(request_payload.get("max_new_tokens", 1400))
                temperature = float(request_payload.get("temperature", 0.0))
                seed = request_payload.get("seed")
                seed = None if seed in {"", None} else int(seed)
                num_images = len(image_paths) if isinstance(image_paths, list) else 0
                preview = image_paths[:2] + image_paths[-2:] if isinstance(image_paths, list) and len(image_paths) > 4 else image_paths
                print(
                    f"[QWEN SERVER] /infer_multi num_images={num_images} "
                    f"max_new_tokens={max_new_tokens} temperature={temperature} "
                    f"seed={seed} images={preview}",
                    flush=True,
                )
                text, resolved_paths = run_multi_image_inference(
                    image_paths=image_paths,
                    system_prompt=str(system_prompt) if system_prompt is not None else None,
                    user_prompt=str(user_prompt),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                return self._send_json(
                    {
                        "ok": True,
                        "text": text,
                        "raw_text": text,
                        "num_images": len(resolved_paths),
                        "image_paths": resolved_paths,
                        "generation": {
                            "seed": seed,
                            "temperature": temperature,
                            "do_sample": bool(float(temperature) != 0.0),
                            "max_new_tokens": max_new_tokens,
                            "model_path": MODEL_PATH,
                            "endpoint": "/infer_multi",
                        },
                    }
                )

            image_path = str(request_payload["image_path"])
            prompt_text = str(request_payload["prompt_text"])
            max_new_tokens = int(request_payload.get("max_new_tokens", 900))
            temperature = float(request_payload.get("temperature", 0.0))
            seed = request_payload.get("seed")
            seed = None if seed in {"", None} else int(seed)

            print(f"[QWEN SERVER] /infer image={image_path}", flush=True)
            text = run_inference(
                image_path=image_path,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )
            output_path = request_payload.get("output_path")
            if output_path:
                out = Path(str(output_path))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
            return self._send_json(
                {
                    "ok": True,
                    "text": text,
                    "raw_text": text,
                    "generation": {
                        "seed": seed,
                        "temperature": temperature,
                        "do_sample": bool(float(temperature) != 0.0),
                        "max_new_tokens": max_new_tokens,
                        "model_path": MODEL_PATH,
                        "endpoint": "/infer",
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            return self._send_json(
                {"ok": False, "error": str(exc), "exception_type": type(exc).__name__},
                code=500,
            )


def create_server(host: str, port: int) -> HTTPServer:
    return HTTPServer((host, int(port)), QwenAnchorRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UPV_VLM_v2 Qwen anchor-selection server.")
    parser.add_argument("--model-path", default="local_models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--load-4bit", action="store_true", help="Load model with bitsandbytes 4-bit quantization.")
    parser.add_argument("--load-8bit", action="store_true", help="Load model with bitsandbytes 8-bit quantization.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Default: auto.")
    parser.add_argument(
        "--max-multi-images",
        type=int,
        default=16,
        help="Safety limit for /infer_multi images per request. Default: 16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MAX_MULTI_IMAGES
    MAX_MULTI_IMAGES = int(args.max_multi_images) if args.max_multi_images else None
    load_model(
        args.model_path,
        load_4bit=bool(args.load_4bit),
        load_8bit=bool(args.load_8bit),
        device_map=str(args.device_map),
    )
    server = create_server(args.host, args.port)
    print(f"[QWEN SERVER] Listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
