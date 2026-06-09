import os
import queue
import threading
import time
import traceback


VIDEO_JOB_QUEUE = queue.Queue()
VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = threading.Lock()
RESULT_TTL_SECONDS = 600
_WORKERS_STARTED = False


def _now():
    return time.time()


def _safe_job_copy(job):
    copied = dict(job)
    result = copied.get("result")
    if isinstance(result, dict):
        copied["result"] = dict(result)
    return copied


def _queue_position(job_id):
    with VIDEO_JOB_QUEUE.mutex:
        queued_ids = [job.get("job_id") for job in list(VIDEO_JOB_QUEUE.queue)]
    try:
        return queued_ids.index(job_id) + 1
    except ValueError:
        return 0


def enqueue_video_job(job):
    job_id = job["job_id"]
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "result": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
            "completed_at": None,
            "result_fetched_at": None,
            "output_path": None,
            "worker": None,
        }
    VIDEO_JOB_QUEUE.put(job)


def update_video_job(job_id, **updates):
    with VIDEO_JOBS_LOCK:
        current = VIDEO_JOBS.setdefault(job_id, {"job_id": job_id})
        current.update(updates)
        current["updated_at"] = _now()


def get_video_job_status(job_id, mark_fetched=True):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            return None
        if mark_fetched and job.get("status") in {"complete", "failed"} and job.get("result_fetched_at") is None:
            job["result_fetched_at"] = _now()
        copied = _safe_job_copy(job)
    copied["queue_position"] = _queue_position(job_id) if copied.get("status") == "queued" else 0
    return copied


def _delete_output_file(output_path, log_func):
    if not output_path:
        return
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
            log_func(f"[WORKER-CLEANUP] Deleted expired output: {output_path}")
    except Exception as exc:
        log_func(f"[WORKER-CLEANUP] Could not delete output {output_path}: {exc}")


def _cleanup_loop(log_func):
    while True:
        time.sleep(30)
        cutoff = _now() - RESULT_TTL_SECONDS
        expired = []
        with VIDEO_JOBS_LOCK:
            for job_id, job in list(VIDEO_JOBS.items()):
                fetched_at = job.get("result_fetched_at")
                if fetched_at and fetched_at < cutoff:
                    expired.append((job_id, job.get("output_path")))
                    del VIDEO_JOBS[job_id]
        for _, output_path in expired:
            _delete_output_file(output_path, log_func)


def _worker_loop(worker_id, process_func, model_path, log_func):
    worker_name = f"WORKER-{worker_id}"
    log_func(f"[{worker_name}] Loading video model: {model_path}")
    from pipeline import _safe_yolo_load
    model = _safe_yolo_load(model_path)
    log_func(f"[{worker_name}] Video model loaded")

    while True:
        job = VIDEO_JOB_QUEUE.get()
        job_id = job["job_id"]
        update_video_job(
            job_id,
            status="processing",
            progress=1,
            message="Processing",
            worker=worker_name,
        )
        log_func(f"[{worker_name}] Starting job_id={job_id}")
        try:
            result = process_func(
                input_path=job["input_path"],
                trim_start=job["trim_start"],
                trim_duration=job["trim_duration"],
                job_id=job_id,
                pole_model=model,
                worker_name=worker_name,
            )
            output_path = _result_output_path(result)
            update_video_job(
                job_id,
                status="complete",
                progress=100,
                message="Complete",
                result=result,
                error=None,
                completed_at=_now(),
                output_path=output_path,
            )
            log_func(f"[{worker_name}] Completed job_id={job_id}")
        except Exception as exc:
            update_video_job(
                job_id,
                status="failed",
                progress=100,
                message="Failed",
                error=str(exc),
                completed_at=_now(),
            )
            log_func(f"[{worker_name}] Failed job_id={job_id}: {exc}")
            log_func(traceback.format_exc())
        finally:
            VIDEO_JOB_QUEUE.task_done()


def _result_output_path(result):
    if not isinstance(result, dict):
        return None
    url = result.get("processed_video_url") or result.get("video_url")
    if not url:
        return None
    rel_path = str(url).split("?", 1)[0].lstrip("/")
    return os.path.abspath(rel_path)


def start_video_workers(process_func, model_path, log_func=print, max_workers=None):
    global _WORKERS_STARTED
    with VIDEO_JOBS_LOCK:
        if _WORKERS_STARTED:
            return
        _WORKERS_STARTED = True

    workers = int(max_workers or os.environ.get("MAX_VIDEO_WORKERS", "2") or 2)
    workers = max(1, min(2, workers))
    cleanup = threading.Thread(target=_cleanup_loop, args=(log_func,), daemon=True, name="video-cleanup")
    cleanup.start()

    for worker_id in range(1, workers + 1):
        thread = threading.Thread(
            target=_worker_loop,
            args=(worker_id, process_func, model_path, log_func),
            daemon=True,
            name=f"video-worker-{worker_id}",
        )
        thread.start()
    log_func(f"[WORKER] Started {workers} video worker thread(s)")
