import os
import threading


_DEVICE_LOCK = threading.Lock()
_GPU_ENABLED = os.environ.get("GPU_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def gpu_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_enabled():
    with _DEVICE_LOCK:
        return bool(_GPU_ENABLED and gpu_available())


def inference_device():
    return "cuda" if gpu_enabled() else "cpu"


def set_gpu_enabled(enabled):
    global _GPU_ENABLED
    requested = bool(enabled)
    available = gpu_available()
    with _DEVICE_LOCK:
        _GPU_ENABLED = requested and available
        active = _GPU_ENABLED

    if not active:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "available": available,
        "enabled": active,
        "device": "cuda" if active else "cpu",
    }


def device_status():
    available = gpu_available()
    with _DEVICE_LOCK:
        enabled = bool(_GPU_ENABLED and available)
    return {
        "available": available,
        "enabled": enabled,
        "device": "cuda" if enabled else "cpu",
    }
