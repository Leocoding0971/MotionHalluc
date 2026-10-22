"""Evaluate one or all MotionHalluc tasks with an OpenAI-compatible vision API.

Evaluation only: invoke evaluate.score separately to compute accuracies.
"""

import argparse
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from evaluate.api_client import APIClient, APIError
from evaluate.media import build_content, video_identity, video_path
from evaluate.results import ResultStore, file_hash, fingerprint, implementation_hash, redact, task_logger
from evaluate.tasks import TASK_FILES, build_prompt, load_samples, normalize_answer, sample_path, selected_tasks


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASK_FILES, "all"], default="all")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/MotionHalluc"))
    parser.add_argument("--samples-json", type=Path, help="Custom annotations; single-task runs only")
    parser.add_argument("--video-root", type=Path, default=Path("dataset"))
    parser.add_argument("--api-url", required=True, help="Complete Chat Completions endpoint, including /chat/completions")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="Run root; each task uses its own child directory")
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/.cache"))
    parser.add_argument("--fps", type=float, default=4.0, help="Target FPS over the whole DH/AH video; ignored for TH")
    parser.add_argument("--frame-size", default="448,448", help="JPEG width,height (default: 448,448)")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=3, help="Retries after the first request (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds between retries (default: 2)")
    parser.add_argument("--limit", type=int, help="Only the first N questions in each selected task")
    return parser


def prepare_runs(args, size, secret):
    """Validate task inputs and experiment identities before any paid request."""
    prepared = []
    code_hash = implementation_hash()
    for task in selected_tasks(args.task):
        path = sample_path(task, args.dataset_dir, args.samples_json)
        samples = load_samples(path, task)
        ids = list(samples)[:args.limit] if args.limit else list(samples)
        inputs = {}
        for sid in ids:
            for field in ("v1", "v2"):
                video = video_path(args.video_root, samples[sid][field])
                inputs[str(video)] = video_identity(video)
        config = {
            "schema_version": 1, "task": task, "mode": "base",
            "api_url": args.api_url, "model_name": args.model_name,
            "samples_json": str(path.resolve()), "samples_hash": file_hash(path), "sample_ids": ids,
            "video_root": str(args.video_root.resolve()), "input_files": inputs,
            "fps": None if task == "th" else args.fps, "frame_size": list(size),
            "max_tokens": args.max_tokens, "implementation_hash": code_hash,
            "prompts_hash": fingerprint({sid: build_prompt(task, samples[sid]) for sid in ids}),
        }
        store = ResultStore(args.output_dir / task, config, secret)
        prepared.append((task, samples, ids, store))
    return prepared


def evaluate_task(task, samples, ids, store, client, args, size):
    failures = 0
    run_id = uuid.uuid4().hex
    with task_logger(store.directory, client.api_key) as logger:
        logger.info("task=%s run_id=%s selected=%d already_successful=%d timeout=%s max_retries=%d retry_delay=%s",
                    task, run_id, len(ids), len(store.predictions), args.timeout, args.max_retries, args.retry_delay)
        for sid in ids:
            if sid in store.predictions:
                continue
            prompt = build_prompt(task, samples[sid])
            common = {"task": task, "sample_id": sid, "run_id": run_id, "prompt": prompt}
            try:
                content, frames = build_content(task, samples[sid], prompt, args.video_root,
                                                args.cache_root, args.fps, size)
            except Exception as error:
                failures += 1
                detail = {**common, "timestamp": datetime.now(timezone.utc).isoformat(),
                          "status": "error", "stage": "media", "attempt": 0,
                          "error": f"{type(error).__name__}: {error}"}
                store.record(detail)
                logger.error("task=%s sample_id=%s media failure: %s", task, sid, detail["error"])
                continue
            attempts = args.max_retries + 1
            for attempt in range(1, attempts + 1):
                detail = {**common, "timestamp": datetime.now(timezone.utc).isoformat(), "frames": frames,
                          "attempt": attempt, "max_attempts": attempts, "status": "error", "stage": "api"}
                try:
                    reply = client.complete(content)
                    detail.update(raw_output=reply.text, api=reply.metadata)
                    detail["stage"] = "answer"
                    answer = normalize_answer(reply.text, task)
                    if answer is None:
                        raise ValueError("Response did not contain one unambiguous allowed answer choice")
                except Exception as error:
                    if isinstance(error, APIError):
                        detail["api"] = error.metadata
                    detail["error"] = f"{type(error).__name__}: {error}"
                    store.record(detail)
                    logger.warning("task=%s sample_id=%s attempt=%d/%d failed: %s; raw_output=%r; api=%s",
                                   task, sid, attempt, attempts, detail["error"],
                                   detail.get("raw_output"), detail.get("api", {}))
                    if attempt == attempts:
                        failures += 1
                        logger.error("task=%s sample_id=%s exhausted attempts; continuing to next sample", task, sid)
                    else:
                        time.sleep(args.retry_delay)
                else:
                    detail.update(status="success", answer=answer)
                    store.record(detail, answer)
                    logger.info("task=%s sample_id=%s attempt=%d/%d answer=%s", task, sid, attempt, attempts, answer)
                    break
        logger.info("task=%s saved=%d/%d failed_samples_this_invocation=%d; run evaluate.score separately",
                    task, len(store.predictions), len(ids), failures)
    return failures


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    secret = os.environ.get("API_KEY", "").strip()
    if not secret:
        parser.error("Set the API_KEY environment variable")
    if args.task == "all" and args.samples_json is not None:
        parser.error("--samples-json is only supported with a single --task")
    args.api_url, args.model_name = args.api_url.strip(), args.model_name.strip()
    endpoint = urlsplit(args.api_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc or not endpoint.path.rstrip("/").endswith("/chat/completions"):
        parser.error("--api-url must be a complete HTTP(S) endpoint ending in /chat/completions")
    if not args.model_name:
        parser.error("--model-name must not be empty")
    try:
        size = tuple(int(x) for x in args.frame_size.split(","))
    except ValueError:
        parser.error("--frame-size must be width,height")
    if len(size) != 2 or min(size) <= 0:
        parser.error("--frame-size needs two positive integers")
    for name in ("fps", "timeout", "max_tokens", "limit"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_retries < 0 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--max-retries and --retry-delay must be nonnegative; delay must be finite")
    try:
        prepared = prepare_runs(args, size, secret)
        client = APIClient(args.api_url, secret, args.model_name, args.timeout, args.max_tokens)
        failures = sum(evaluate_task(task, samples, ids, store, client, args, size)
                       for task, samples, ids, store in prepared)
        return 1 if failures else 0
    except Exception as error:
        print(redact(f"Evaluation stopped: {type(error).__name__}: {error}", secret), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
