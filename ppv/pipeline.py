"""Two validated parser requests, local execution, then paired-video verification."""

import json
from pathlib import Path
import time

from ppv.api_client import APIClient
from ppv.media import build_content, sample_video
from ppv.queries import available_stages, parse_json, validate_qf, validate_qk, validate_verdict
from ppv.results import RunRecord


def load_prompt(name):
    return (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()


def request(stage, settings, messages, validator, record):
    client = APIClient(settings)
    label = {"qf": "Qf", "qk": "Qk", "verification": "Verification"}[stage]
    for attempt in range(1, settings["max_attempts"] + 1):
        if attempt == 1:
            print(f"[{label}] Request", flush=True)
        else:
            print(f"[{label}] Retry {attempt - 1}/{settings['max_attempts'] - 1}", flush=True)
        record.log("model_request", stage=stage, attempt=attempt, model=settings["model_name"],
                   endpoint=settings["api_url"], timeout=settings["timeout"],
                   max_tokens=settings["max_tokens"], messages=messages)
        try:
            raw, metadata = client.complete(messages)
            record.log("model_response", stage=stage, attempt=attempt, raw=raw, metadata=metadata)
            parsed = parse_json(raw)
            normalized = validator(parsed)
        except Exception as error:
            # Retry the same request; never repair with an unchecked or default query.
            retry = attempt < settings["max_attempts"]
            record.log("attempt_failed", stage=stage, attempt=attempt, error_type=type(error).__name__,
                       error=str(error), metadata=getattr(error, "metadata", {}), retry=retry,
                       retry_delay=settings["retry_delay"] if retry else 0)
            if not retry:
                raise RuntimeError(f"{stage} failed after {attempt} attempts: {error}") from None
            if settings["retry_delay"]:
                time.sleep(settings["retry_delay"])
        else:
            result = {"raw_output": raw, "parsed_output": parsed, "query": normalized, "attempts": attempt}
            record.log("request_succeeded", stage=stage, **result)
            return result


def parser_messages(prompt_name, inputs):
    return [{"role": "system", "content": load_prompt(prompt_name)},
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)}]


def run(config):
    record = RunRecord(config["output_dir"], [config[role]["api_key"]
                       for role in ("parser", "verifier") if role in config])
    phase = "input_validation"
    record.log("run_started", parameters=config)
    record.save(parameters=config)
    print(f"[{config['mode'].upper()}] Starting", flush=True)
    try:
        inputs = config["input"]
        videos, video_info = [], {}
        for key in ("a", "b"):
            images, metadata = sample_video(inputs[f"video_{key}"], **config["sampling"])
            videos.append(images)
            video_info[key] = metadata
            record.log("video_loaded", video=key, metadata=metadata)
        if config["mode"] == "ppv":
            from ppv.motion import check_correspondence, execute, load_motion

            arrays = {}
            for key in ("a", "b"):
                arrays[key], metadata = load_motion(inputs[f"motion_{key}"])
                correspondence = check_correspondence(video_info[key]["decoded_frames"],
                                                       len(arrays[key]), config["max_frame_difference"])
                record.log("motion_loaded", video=key, metadata=metadata,
                           correspondence=correspondence)
            common = {"action_type": inputs["action_name"], "action_description": inputs["action_description"]}
            phase = "qf"
            qf = request(phase, config["parser"], parser_messages("frame_localization", {
                **common, "stage_descriptions": available_stages(inputs),
                "correction": inputs["correction"]}), lambda value: validate_qf(value, inputs), record)
            record.save(qf=qf)
            phase = "qk"
            qk = request(phase, config["parser"], parser_messages("instruction_measurements", {
                **common, "corrections": [inputs["correction"]]}), lambda value: validate_qk(value, inputs), record)
            record.save(qk=qk)
            phase = "execution"
            execution = {}
            for key in ("a", "b"):
                execution[key] = execute(arrays[key], qf["query"], qk["query"])
                record.log("execution_succeeded", video=key, execution=execution[key])
                record.save(execution=execution)
            report = {"report_video1": execution["a"]["report"], "report_video2": execution["b"]["report"]}
            prompt = load_prompt("verification_ppv").format(
                correction=inputs["correction"], report=json.dumps(report, ensure_ascii=False, indent=2))
        else:
            prompt = load_prompt("verification_base").format(correction=inputs["correction"])
        phase = "verification"
        verdict = request(phase, config["verifier"], [{"role": "user", "content": build_content(videos, prompt)}],
                          validate_verdict, record)
        # A verdict has reasoning/result rather than a query; keep that clear in the public artifact.
        verdict.update(verdict.pop("query"))
        record.save(status="success", verification=verdict, finished_at=record.now())
        record.log("run_succeeded", verification=verdict)
    except (Exception, KeyboardInterrupt) as error:
        record.log("run_failed", phase=phase, error_type=type(error).__name__, error=str(error))
        record.save(status="failed", failure={"phase": phase, "type": type(error).__name__, "reason": str(error)},
                    finished_at=record.now())
    return record.directory, record.result["status"] == "success"
