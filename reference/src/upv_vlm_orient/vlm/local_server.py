from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

from upv_vlm_orient.vlm.prompt_registry import get_prompt, list_prompt_versions
from upv_vlm_orient.vlm.qwen_runtime import QwenRuntime, extract_json


class QwenVlmHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address, runtime: QwenRuntime):
        super().__init__(server_address, QwenVlmRequestHandler)
        self.runtime = runtime


class QwenVlmRequestHandler(BaseHTTPRequestHandler):
    server: QwenVlmHttpServer

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/health":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        self._send_json(
            {
                "ok": True,
                "ready": self.server.runtime.ready,
                "runtime": self.server.runtime.metadata(),
                "prompt_versions": list_prompt_versions(),
            }
        )

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/infer":
            self._handle_infer()
            return
        if parsed_path.path == "/shutdown":
            self._send_json({"ok": True, "shutdown": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def _handle_infer(self) -> None:
        try:
            request = self._read_json_body()
            result = run_infer_request(runtime=self.server.runtime, request=request)
            self._send_json(result, status=200 if result["ok"] else 400)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_infer_request(runtime: QwenRuntime, request: dict[str, Any]) -> dict:
    image_path = request.get("image_path")
    prompt_text = request.get("prompt_text")
    prompt_version = request.get("prompt_version") or "current_default"
    max_new_tokens = int(request.get("max_new_tokens") or 700)
    temperature = float(request.get("temperature") or 0.0)
    output_path = request.get("output_path")

    if not image_path:
            return {
                "ok": False,
                "image_path": image_path,
                "prompt_version": prompt_version,
                "raw_response": None,
                "parsed_json": None,
                "parse_ok": False,
                "parse_error": "Missing required field: image_path",
                "output_path": output_path,
                "error": "Missing required field: image_path",
            }

    resolved_prompt_version: str | None
    if prompt_text:
        resolved_prompt_version = prompt_version if prompt_version else None
    else:
        resolved_prompt_version, prompt_text = get_prompt(prompt_version)

    try:
        raw_response = runtime.infer(
            image_path=image_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        try:
            parsed_json = extract_json(raw_response)
            parse_ok = True
            parse_error = None
        except Exception as exc:
            parsed_json = None
            parse_ok = False
            parse_error = str(exc)

        result = {
            "ok": True,
            "image_path": str(image_path),
            "prompt_version": resolved_prompt_version,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
            "parse_ok": parse_ok,
            "output_path": output_path,
            "error": None,
            "parse_error": parse_error,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "image_path": str(image_path),
            "prompt_version": resolved_prompt_version,
            "raw_response": None,
            "parsed_json": None,
            "parse_ok": False,
            "parse_error": str(exc),
            "output_path": output_path,
            "error": str(exc),
        }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def serve_qwen_vlm(
    host: str,
    port: int,
    model_dir: str,
    load_4bit: bool,
    mock: bool,
) -> None:
    runtime = QwenRuntime(model_dir=model_dir, load_4bit=load_4bit, mock=mock)
    runtime.load()
    server = QwenVlmHttpServer((host, port), runtime=runtime)
    print(f"qwen_vlm_server listening on http://{host}:{port} mock={mock}", flush=True)
    server.serve_forever()
