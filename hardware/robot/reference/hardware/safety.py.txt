from __future__ import annotations


def require_execute_confirmation(confirm: str | None, expected: str) -> None:
    if confirm != expected:
        raise PermissionError(f"execute mode requires confirmation text: {expected}")


def validate_execution_request(
    *,
    backend: str,
    confirm: str | None,
    expected_confirm: str,
    allow_real_hardware: bool,
) -> dict:
    normalized = str(backend or "none").lower()
    if normalized == "none":
        return {"ok": False, "failure_reason": "execution_backend_none"}
    if normalized == "simulated":
        if confirm != expected_confirm:
            return {"ok": False, "failure_reason": f"simulated execute requires confirm={expected_confirm}"}
        return {"ok": True, "backend": "simulated", "hardware_allowed": False}
    if normalized == "real":
        if confirm != expected_confirm:
            return {"ok": False, "failure_reason": f"real execute requires confirm={expected_confirm}"}
        if not allow_real_hardware:
            return {"ok": False, "failure_reason": "real execute requires --allow-real-hardware"}
        return {"ok": True, "backend": "real", "hardware_allowed": True}
    return {"ok": False, "failure_reason": f"unsupported execution backend: {backend}"}
