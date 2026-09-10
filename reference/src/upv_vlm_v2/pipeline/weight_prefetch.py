"""Optional bounded-buffer OS page-cache warming; never allocates GPU memory."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


@contextmanager
def prefetch_during_perception(config, output_dir, dry_run=False):
    settings = config.get("managed_qwen", {})
    if dry_run or not settings.get("enabled") or not settings.get("prefetch_weights"):
        yield
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"method": "sequential_read_into_os_page_cache", "gpu_memory_used": False,
              "bytes_read": 0, "completed": False, "cache_residency_guaranteed": False}
    cancel = threading.Event()
    thread = None
    start = time.perf_counter()

    def read_weights(paths, reserve):
        began = time.perf_counter()
        try:
            buffer = bytearray(8 * 1024 * 1024)
            for path in paths:
                with path.open("rb", buffering=0) as handle:
                    while True:
                        if cancel.is_set():
                            record["stopped_reason"] = "perception_failed_or_interrupted"
                            return
                        if available_memory() < reserve:
                            record["stopped_reason"] = "available_memory_below_reserve"
                            return
                        n = handle.readinto(buffer)
                        if not n:
                            break
                        record["bytes_read"] += n
            record["completed"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["read_ms"] = (time.perf_counter() - began) * 1000

    try:
        try:
            model = Path(settings["model_path"])
            index = json.loads((model / "model.safetensors.index.json").read_text())
            paths = [(model / name).resolve() for name in sorted(set(index["weight_map"].values()))]
            if not all(p.is_relative_to(model.resolve()) for p in paths):
                raise ValueError("shard path escapes model directory")
            total = sum(p.stat().st_size for p in paths)
            reserve = int(float(settings.get("prefetch_reserve_gib", 8)) * 2**30)
            memory = available_memory()
            record.update(weight_bytes=total, available_memory_before=memory, reserve_bytes=reserve)
            if memory < total + reserve:
                record["stopped_reason"] = "insufficient_available_memory_for_weights_plus_reserve"
            else:
                thread = threading.Thread(target=read_weights, args=(paths, reserve), name="qwen-weight-prefetch")
                thread.start()
                print("Prefetching Qwen weights into CPU page cache during perception.", flush=True)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        try:
            yield
        except BaseException:
            cancel.set()
            raise
    finally:
        wait = time.perf_counter()
        if thread is not None:
            thread.join()
        record["wait_after_perception_ms"] = (time.perf_counter() - wait) * 1000
        record["context_wall_ms"] = (time.perf_counter() - start) * 1000
        (out / "weight_prefetch.json").write_text(json.dumps(record, indent=2) + "\n")
