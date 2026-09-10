"""Long-running Llama vision anchor server for UPV_VLM_v2.

This server intentionally mirrors the Qwen anchor server HTTP API while using
the correct Mllama model family for Llama-3.2-Vision checkpoints.
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
MODEL_TYPE: str | None = None
MODEL_CLASS: str | None = None
PROCESSOR_CLASS: str | None = None
DEVICE: str | None = None
MAX_MULTI_IMAGES: int | None = None


def _read_model_type(model_path: str | Path) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = data.get("model_type")
    return str(model_type) if model_type is not None else None


def _cuda_memory() -> dict[str, float | None]:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {"cuda_memory_free_mib": None, "cuda_memory_total_mib": None}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_memory_free_mib": round(float(free) / (1024.0 * 1024.0), 3),
            "cuda_memory_total_mib": round(float(total) / (1024.0 * 1024.0), 3),
        }
    except Exception:
        return {"cuda_memory_free_mib": None, "cuda_memory_total_mib": None}


def load_model(
    model_path: str,
    load_4bit: bool = False,
    load_8bit: bool = False,
    device_map: str = "auto",
) -> None:
    """Load a Llama vision model once at server startup."""

    global MODEL, PROCESSOR, MODEL_PATH, MODEL_TYPE, MODEL_CLASS, PROCESSOR_CLASS, DEVICE

    print(f"[LLAMA VISION SERVER] Loading model: {model_path}", flush=True)
    MODEL_TYPE = _read_model_type(model_path)
    print(f"[LLAMA VISION SERVER] config model_type={MODEL_TYPE}", flush=True)
    if MODEL_TYPE != "mllama":
        raise ValueError(f"run_llama_vision_anchor_server only accepts model_type=mllama, got {MODEL_TYPE!r}")

    import torch  # type: ignore
    from transformers import AutoProcessor, BitsAndBytesConfig  # type: ignore

    try:
        from transformers import MllamaForConditionalGeneration as ModelClass  # type: ignore
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as ModelClass  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Installed transformers does not expose MllamaForConditionalGeneration "
                "or AutoModelForVision2Seq; cannot safely load Llama vision."
            ) from exc

    if load_4bit and load_8bit:
        raise ValueError("Use only one of --load-4bit or --load-8bit, not both.")

    cuda = bool(torch.cuda.is_available())
    dtype = torch.bfloat16 if cuda else torch.float32
    print(
        f"[LLAMA VISION SERVER] cuda={cuda} dtype={dtype} "
        f"load_4bit={load_4bit} load_8bit={load_8bit} device_map={device_map}",
        flush=True,
    )
    print(f"[LLAMA VISION SERVER] memory_before_load={_cuda_memory()}", flush=True)

    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    PROCESSOR_CLASS = type(PROCESSOR).__name__

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

    MODEL = ModelClass.from_pretrained(model_path, **model_kwargs)
    MODEL_CLASS = type(MODEL).__name__
    if "Qwen" in MODEL_CLASS:
        raise RuntimeError(f"Invalid model class for Llama vision server: {MODEL_CLASS}")

    if not cuda:
        MODEL.to("cpu")

    MODEL.eval()
    MODEL_PATH = str(model_path)
    DEVICE = "cuda" if cuda else "cpu"
    print(
        f"[LLAMA VISION SERVER] Model loaded. model_class={MODEL_CLASS} "
        f"processor_class={PROCESSOR_CLASS} device={DEVICE}",
        flush=True,
    )
    print(f"[LLAMA VISION SERVER] memory_after_load={_cuda_memory()}", flush=True)


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


def _resolve_existing_images(image_paths: list[Any]) -> list[Path]:
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths must be a non-empty list.")
    if MAX_MULTI_IMAGES is not None and len(image_paths) > int(MAX_MULTI_IMAGES):
        raise ValueError(f"image_paths length {len(image_paths)} exceeds max_multi_images={MAX_MULTI_IMAGES}.")

    resolved_images: list[Path] = []
    for idx, image_path in enumerate(image_paths, start=1):
        resolved = Path(str(image_path)).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"image_paths[{idx - 1}] not found: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"image_paths[{idx - 1}] is not a file: {resolved}")
        resolved_images.append(resolved)
    return resolved_images


def _move_inputs_to_device(inputs: Any) -> Any:
    target_device = _model_device()
    try:
        return inputs.to(target_device)
    except Exception:
        return {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _build_chat_text(*, image_count: int, user_prompt: str, system_prompt: str | None = None) -> str:
    if PROCESSOR is None:
        raise RuntimeError("Llama vision processor is not loaded.")
    if not str(user_prompt).strip():
        raise ValueError("user_prompt or prompt_text is required.")

    user_content: list[dict[str, str]] = [{"type": "image"} for _ in range(image_count)]
    merged_prompt = str(user_prompt)
    messages: list[dict[str, Any]]
    if system_prompt and str(system_prompt).strip():
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": [*user_content, {"type": "text", "text": merged_prompt}]},
        ]
    else:
        messages = [{"role": "user", "content": [*user_content, {"type": "text", "text": merged_prompt}]}]

    try:
        return PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if system_prompt and str(system_prompt).strip():
            merged_prompt = f"{system_prompt}\n\n{user_prompt}"
        fallback_messages = [{"role": "user", "content": [*user_content, {"type": "text", "text": merged_prompt}]}]
        return PROCESSOR.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=True)


def _generate_from_images(
    *,
    image_paths: list[Any],
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 900,
    temperature: float = 0.0,
) -> tuple[str, list[str]]:
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Llama vision model is not loaded.")

    import torch  # type: ignore
    from PIL import Image

    resolved_images = _resolve_existing_images(image_paths)
    images = [Image.open(path).convert("RGB") for path in resolved_images]
    text = _build_chat_text(image_count=len(images), user_prompt=user_prompt, system_prompt=system_prompt)
    inputs = PROCESSOR(images=images, text=text, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs)

    print(f"[LLAMA VISION SERVER] memory_before_infer={_cuda_memory()}", flush=True)
    with torch.inference_mode():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False if float(temperature) == 0.0 else True,
            temperature=None if float(temperature) == 0.0 else float(temperature),
        )
    input_len = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_len:]
    text_out = PROCESSOR.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f"[LLAMA VISION SERVER] memory_after_infer={_cuda_memory()}", flush=True)
    return text_out, [str(path) for path in resolved_images]


def run_inference(image_path: str, prompt_text: str, max_new_tokens: int = 900, temperature: float = 0.0) -> str:
    text, _paths = _generate_from_images(
        image_paths=[image_path],
        user_prompt=prompt_text,
        system_prompt=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return text


def run_multi_image_inference(
    image_paths: list[Any],
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1600,
    temperature: float = 0.0,
) -> tuple[str, list[str]]:
    return _generate_from_images(
        image_paths=image_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


class LlamaVisionAnchorRequestHandler(BaseHTTPRequestHandler):
    server_version = "UPV_VLM_v2_LlamaVisionAnchorServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[LLAMA VISION SERVER] {self.address_string()} - {fmt % args}", flush=True)

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
                    "model_family": "llama_vision",
                    "model_type": MODEL_TYPE,
                    "model_class": MODEL_CLASS,
                    "processor_class": PROCESSOR_CLASS,
                    "device": DEVICE,
                    **_cuda_memory(),
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
                max_new_tokens = int(request_payload.get("max_new_tokens", 1600))
                temperature = float(request_payload.get("temperature", 0.0))
                num_images = len(image_paths) if isinstance(image_paths, list) else 0
                preview = image_paths[:2] + image_paths[-2:] if isinstance(image_paths, list) and len(image_paths) > 4 else image_paths
                print(
                    f"[LLAMA VISION SERVER] /infer_multi num_images={num_images} "
                    f"max_new_tokens={max_new_tokens} temperature={temperature} images={preview}",
                    flush=True,
                )
                text, resolved_paths = run_multi_image_inference(
                    image_paths=image_paths,
                    system_prompt=str(system_prompt) if system_prompt is not None else None,
                    user_prompt=str(user_prompt),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return self._send_json(
                    {
                        "ok": True,
                        "text": text,
                        "raw_text": text,
                        "num_images": len(resolved_paths),
                        "image_paths": resolved_paths,
                        "model_family": "llama_vision",
                    }
                )

            image_path = str(request_payload["image_path"])
            prompt_text = str(request_payload["prompt_text"])
            max_new_tokens = int(request_payload.get("max_new_tokens", 900))
            temperature = float(request_payload.get("temperature", 0.0))
            print(f"[LLAMA VISION SERVER] /infer image={image_path}", flush=True)
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
    return HTTPServer((host, int(port)), LlamaVisionAnchorRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UPV_VLM_v2 Llama vision anchor server.")
    parser.add_argument("--model-path", default="local_models/Llama-3.2-11B-Vision-Instruct")
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
    print(f"[LLAMA VISION SERVER] Listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

