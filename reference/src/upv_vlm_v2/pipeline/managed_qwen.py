"""Opt-in process-owned Qwen residency; never attach to or kill another server."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def validate_managed_qwen(config):
    settings = config.get("managed_qwen", {})
    if not settings.get("enabled"):
        return
    anchor = config["anchor_selection"]
    port = int(settings.get("port", 8896))
    if anchor.get("qwen_server_url", "").rstrip("/") != f"http://127.0.0.1:{port}" or anchor.get("qwen_server_endpoint") != "/infer_multi":
        raise ValueError("Managed Qwen requires its local /infer_multi endpoint")
    for key in ("launcher", "python", "model_path"):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(settings[key])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))  # Fail before perception or motion if occupied.


@contextmanager
def managed_qwen(config, output_dir, dry_run=False):
    settings = config.get("managed_qwen", {})
    if dry_run or not settings.get("enabled"):
        yield
        return
    validate_managed_qwen(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = [settings.get("python", sys.executable), "-u", settings["launcher"],
               "--model-path", settings["model_path"], "--port", str(settings.get("port", 8896))]
    record = {"command": command, "model_path": settings["model_path"],
              "model_loading_included_in_lifecycle": True, "success": False}
    start = time.perf_counter()
    proc = None
    try:
        with (out / "managed_qwen_server.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            record["pid"] = proc.pid
            deadline = start + float(settings.get("startup_timeout_sec", 180))
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Qwen exited during startup; inspect managed_qwen_server.log")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Qwen startup timed out")
                try:
                    with urlopen(config["anchor_selection"]["qwen_server_url"] + "/health", timeout=1) as response:
                        health = json.load(response)
                    if health.get("model_loaded") and Path(health.get("model_path", "")).resolve() == Path(settings["model_path"]).resolve():
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            record["startup_to_health_ms"] = (time.perf_counter() - start) * 1000
            print("Perception finished; managed Qwen ready for /infer_multi.", flush=True)
            yield
            record["success"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop = time.perf_counter()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        record["shutdown_ms"] = (time.perf_counter() - stop) * 1000
        record["lifecycle_ms"] = (time.perf_counter() - start) * 1000
        (out / "managed_qwen_lifecycle.json").write_text(json.dumps(record, indent=2) + "\n")
        print("Managed Qwen stopped before path planning/robot execution.", flush=True)
