"""Load all run settings from one JSON file; paths are relative to that file."""

import json
import math
from pathlib import Path


def positive(value, name, integer=False, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def model_settings(value, role):
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    settings = {key: text(value.get(key), f"{role}.{key}")
                for key in ("api_key", "api_url", "model_name")}
    settings["timeout"] = positive(value.get("timeout", 120 if role == "parser" else 600), f"{role}.timeout")
    settings["max_tokens"] = value.get("max_tokens", None if role == "parser" else 3000)
    if settings["max_tokens"] is not None:
        positive(settings["max_tokens"], f"{role}.max_tokens", integer=True)
    settings["max_attempts"] = positive(value.get("max_attempts", 10), f"{role}.max_attempts", integer=True)
    settings["retry_delay"] = positive(value.get("retry_delay", 2), f"{role}.retry_delay", allow_zero=True)
    return settings


def read_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    mode = raw.get("mode", "ppv")
    if mode not in ("ppv", "base"):
        raise ValueError("mode must be ppv or base")
    source = raw.get("input")
    if not isinstance(source, dict):
        raise ValueError("input must be an object")
    inputs = {key: text(source.get(key), f"input.{key}") for key in
              ("action_name", "action_description", "correction")}

    def resolve(value, name):
        p = Path(text(value, name)).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())

    for key in ("video_a", "video_b"):
        inputs[key] = resolve(source.get(key), f"input.{key}")
    config = {"mode": mode, "input": inputs, "verifier": model_settings(raw.get("verifier"), "verifier")}
    if mode == "ppv":
        stages = source.get("stage_descriptions")
        if not isinstance(stages, list) or not stages:
            raise ValueError("input.stage_descriptions must be a nonempty list")
        inputs["stage_descriptions"] = [text(item, "stage description") for item in stages]
        for key in ("motion_a", "motion_b"):
            inputs[key] = resolve(source.get(key), f"input.{key}")
        config["parser"] = model_settings(raw.get("parser"), "parser")
        config["max_frame_difference"] = positive(raw.get("max_frame_difference", 10),
                                                  "max_frame_difference", integer=True, allow_zero=True)
    # Base deliberately ignores parser, motion paths, stages, and correspondence settings.
    sampling = raw.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    size = sampling.get("frame_size", [448, 448])
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("sampling.frame_size must be [width, height]")
    config["sampling"] = {"fps": positive(sampling.get("fps", 4), "sampling.fps"),
                          "frame_size": [positive(x, "frame size", integer=True) for x in size]}
    config["output_dir"] = resolve(raw.get("output_dir", "../outputs/ppv"), "output_dir")
    return config
