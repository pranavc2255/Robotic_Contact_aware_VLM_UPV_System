"""Persistent Penguin-VL server for saved single- and multi-image anchor audits."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


MODEL: Any | None = None
PROCESSOR: Any | None = None
MODEL_PATH: str | None = None
DEVICE: str | None = None
DTYPE: Any | None = None
TRANSFORMERS_VERSION: str | None = None
VISION_ATTENTION_BACKEND: str | None = None
MAX_MULTI_IMAGES = 16

# Penguin uses trusted local custom code, which Transformers copies into its
# dynamic-module cache. The default home cache is read-only on this machine.
os.environ.setdefault("HF_HOME", "/tmp/upv_vlm_huggingface")


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser() / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_penguin_model_path(model_path: str | Path) -> str:
    model_type = str(read_model_config(model_path).get("model_type") or "")
    if model_type != "penguinvl_qwen3":
        raise ValueError(f"Expected model_type='penguinvl_qwen3', got {model_type!r}: {model_path}")
    return model_type


def _torch_sdpa_varlen_func(
    query: Any,
    key: Any,
    value: Any,
    *,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: float | None = None,
    **_: Any,
) -> Any:
    """FlashAttention varlen-compatible fallback using PyTorch SDPA.

    Penguin's vision encoder packs multiple images into one token tensor and
    supplies cumulative sequence lengths. PyTorch SDPA does not accept that
    packed representation directly, so each image segment is evaluated
    independently and the outputs are concatenated back into packed order.
    """
    del max_seqlen_q, max_seqlen_k
    import torch
    import torch.nn.functional as functional

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Expected packed query/key/value tensors with shape [tokens, heads, head_dim].")
    if len(cu_seqlens_q) != len(cu_seqlens_k):
        raise ValueError("Query and key cumulative sequence arrays must have equal lengths.")

    segments: list[Any] = []
    for index in range(len(cu_seqlens_q) - 1):
        q_start = int(cu_seqlens_q[index].item())
        q_end = int(cu_seqlens_q[index + 1].item())
        k_start = int(cu_seqlens_k[index].item())
        k_end = int(cu_seqlens_k[index + 1].item())
        q = query[q_start:q_end]
        k = key[k_start:k_end]
        v = value[k_start:k_end]
        if q.numel() == 0 or k.numel() == 0:
            raise ValueError(f"Empty packed attention segment at index {index}.")

        # Penguin uses grouped-query attention (16 query heads and 8 KV heads).
        # Repeat KV heads explicitly for compatibility across PyTorch versions.
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1] != 0 or k.shape[1] != v.shape[1]:
                raise ValueError(
                    f"Unsupported query/KV head counts: q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}."
                )
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # SDPA expects [batch, heads, sequence, head_dim].
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        output = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(dropout_p),
            is_causal=bool(causal),
            scale=softmax_scale,
        )
        segments.append(output.squeeze(0).transpose(0, 1))

    if not segments:
        return torch.empty_like(query)
    return torch.cat(segments, dim=0)


def install_penguin_vision_attention_backend() -> str:
    """Provide the function hard-coded by Penguin's custom vision encoder."""
    global VISION_ATTENTION_BACKEND

    encoder_modules = [
        module
        for name, module in sys.modules.items()
        if name.endswith("modeling_penguinvl_encoder") and module is not None
    ]
    if not encoder_modules:
        # The trusted module normally has a cache-qualified package name. This
        # import is a fallback for environments that load it as a local module.
        try:
            encoder_modules = [importlib.import_module("modeling_penguinvl_encoder")]
        except ModuleNotFoundError as exc:
            raise RuntimeError("Loaded Penguin vision encoder module was not found.") from exc

    try:
        from flash_attn import flash_attn_varlen_func as attention_func  # type: ignore

        backend = "flash_attention_2"
    except (ImportError, ModuleNotFoundError):
        attention_func = _torch_sdpa_varlen_func
        backend = "torch_sdpa_varlen_fallback"

    for module in encoder_modules:
        setattr(module, "flash_attn_varlen_func", attention_func)
    VISION_ATTENTION_BACKEND = backend
    print(f"[PENGUIN SERVER] Vision attention backend: {backend}", flush=True)
    return backend


def load_model(model_path: str, device_map: str = "auto") -> None:
    global MODEL, PROCESSOR, MODEL_PATH, DEVICE, DTYPE, TRANSFORMERS_VERSION
    ensure_penguin_model_path(model_path)
    import torch  # type: ignore
    import transformers  # type: ignore
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

    cuda = bool(torch.cuda.is_available())
    DTYPE = torch.bfloat16 if cuda else torch.float32
    TRANSFORMERS_VERSION = getattr(transformers, "__version__", "unknown")
    print(
        f"[PENGUIN SERVER] Loading {model_path}; transformers={TRANSFORMERS_VERSION} "
        f"cuda={cuda} dtype={DTYPE} device_map={device_map}",
        flush=True,
    )
    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map if cuda else None,
        torch_dtype=DTYPE,
    )
    install_penguin_vision_attention_backend()
    if not cuda:
        MODEL.to("cpu")
    MODEL.eval()
    MODEL_PATH = str(model_path)
    DEVICE = "cuda" if cuda else "cpu"
    print("[PENGUIN SERVER] Model loaded.", flush=True)


def _model_device() -> Any:
    if MODEL is None:
        return "cpu"
    device = getattr(MODEL, "device", None)
    if device is not None:
        return device
    return next(MODEL.parameters()).device


def _resolve_images(values: Any) -> list[Path]:
    if not isinstance(values, list) or not values:
        raise ValueError("image_paths must be a non-empty list.")
    if len(values) > MAX_MULTI_IMAGES:
        raise ValueError(f"image_paths length {len(values)} exceeds max_multi_images={MAX_MULTI_IMAGES}.")
    paths: list[Path] = []
    for idx, value in enumerate(values):
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image_paths[{idx}] not found: {path}")
        paths.append(path)
    return paths


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    import torch  # type: ignore

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_multi_image_inference(
    image_paths: Any,
    user_prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 1400,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[str, list[str]]:
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Penguin-VL model is not loaded.")
    import torch  # type: ignore

    paths = _resolve_images(image_paths)
    if not str(user_prompt).strip():
        raise ValueError("user_prompt is required for /infer_multi.")
    conversation: list[dict[str, Any]] = []
    if system_prompt and str(system_prompt).strip():
        conversation.append({"role": "system", "content": str(system_prompt)})
    content: list[dict[str, Any]] = [
        {"type": "image", "image": {"image_path": str(path)}} for path in paths
    ]
    content.append({"type": "text", "text": str(user_prompt)})
    conversation.append({"role": "user", "content": content})

    inputs = PROCESSOR(conversation=conversation, return_tensors="pt")
    device = _model_device()
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    if "pixel_values" in inputs and DTYPE is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    _apply_seed(seed)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) != 0.0,
    }
    if float(temperature) != 0.0:
        generation_kwargs["temperature"] = float(temperature)
    with torch.inference_mode():
        output_ids = MODEL.generate(**inputs, **generation_kwargs)
    text = PROCESSOR.decode(output_ids[0], skip_special_tokens=True)
    return text, [str(path) for path in paths]


class PenguinAnchorRequestHandler(BaseHTTPRequestHandler):
    server_version = "UPV_VLM_v2_PenguinVLAnchorServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[PENGUIN SERVER] {self.address_string()} - {fmt % args}", flush=True)

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
                    "server": "penguin_vl_anchor_server",
                    "model_loaded": MODEL is not None and PROCESSOR is not None,
                    "model_path": MODEL_PATH,
                    "device": DEVICE,
                    "transformers_version": TRANSFORMERS_VERSION,
                    "vision_attention_backend": VISION_ATTENTION_BACKEND,
                }
            )
        return self._send_json({"ok": False, "error": "not_found"}, code=404)

    def do_POST(self) -> None:
        endpoint = self.path.rstrip("/")
        if endpoint not in {"/infer", "/infer_multi"}:
            return self._send_json({"ok": False, "error": "expected /infer or /infer_multi"}, code=404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            paths = payload.get("image_paths")
            if endpoint == "/infer":
                paths = [payload.get("image_path") or payload.get("image")]
            seed_value = payload.get("seed")
            seed = None if seed_value in {None, ""} else int(seed_value)
            text, resolved = run_multi_image_inference(
                image_paths=paths,
                system_prompt=payload.get("system_prompt"),
                user_prompt=str(payload.get("user_prompt") or payload.get("prompt_text") or payload.get("prompt") or ""),
                max_new_tokens=int(payload.get("max_new_tokens", 1400)),
                temperature=float(payload.get("temperature", 0.0)),
                seed=seed,
            )
            return self._send_json(
                {
                    "ok": True,
                    "text": text,
                    "raw_text": text,
                    "num_images": len(resolved),
                    "image_paths": resolved,
                    "generation": {"model_path": MODEL_PATH, "endpoint": endpoint, "seed": seed},
                }
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._send_json(
                {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}, code=500
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Penguin-VL anchor-selection server.")
    parser.add_argument("--model-path", default="local_models/Penguin-VL-2B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-multi-images", type=int, default=16)
    parser.add_argument("--check-only", action="store_true", help="Validate config/import support without loading weights.")
    return parser.parse_args()


def main() -> None:
    global MAX_MULTI_IMAGES
    args = parse_args()
    MAX_MULTI_IMAGES = int(args.max_multi_images)
    ensure_penguin_model_path(args.model_path)
    import transformers  # type: ignore

    support = {
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "AutoModelForCausalLM": hasattr(transformers, "AutoModelForCausalLM"),
        "AutoProcessor": hasattr(transformers, "AutoProcessor"),
    }
    print(f"[PENGUIN SERVER] support={json.dumps(support)}", flush=True)
    if args.check_only:
        print("[PENGUIN SERVER] check-only passed; model weights were not loaded.", flush=True)
        return
    load_model(args.model_path, device_map=str(args.device_map))
    server = HTTPServer((args.host, int(args.port)), PenguinAnchorRequestHandler)
    print(f"[PENGUIN SERVER] Listening at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
