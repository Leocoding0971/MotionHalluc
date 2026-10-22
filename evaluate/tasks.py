"""Benchmark loading and base prompts from the original Qwen API workers.

Prompt sources: eval/api_model/qwen/parallel_binary/binaryqa.py,
parallel_mcq/mcq.py, and parallel_clipqa/clipqa.py in the research repository.
The rendered base text and image-group labels are preserved verbatim.
"""

import re
from pathlib import Path

from evaluate.results import read_json

TASK_FILES = {"dh": "DH.json", "ah1": "AH1.json", "ah2": "AH2.json", "th": "TH.json"}
VIDEO_LABELS = ("[VIDEO 1: USER PERFORMANCE]", "[VIDEO 2: REFERENCE PERFORMANCE]")
CLIP_GROUPS = (
    ("[VIDEO 1: QUERY CLIP]", "v1", "f1"),
    ("[VIDEO 2 CANDIDATE A]", "v2", "f2_A"),
    ("[VIDEO 2 CANDIDATE B]", "v2", "f2_B"),
    ("[VIDEO 2 CANDIDATE C]", "v2", "f2_C"),
)


def selected_tasks(task):
    return list(TASK_FILES) if task == "all" else [task]


def sample_path(task, dataset_dir, override=None):
    return Path(override) if override is not None else Path(dataset_dir) / TASK_FILES[task]


def choices(task):
    return "ABC" if task == "th" else "AB"


def load_samples(path, task):
    samples = read_json(path)
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"{path}: expected a nonempty object keyed by sample ID")
    for sid, sample in samples.items():
        if not sid.strip() or not isinstance(sample, dict):
            raise ValueError(f"{path}: invalid sample_id={sid!r}")
        for field in ("v1", "v2", "q"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"sample_id={sid}: {field} must be a nonempty string")
        if task in ("ah1", "ah2"):
            options = sample.get("c")
            if not isinstance(options, list) or len(options) != 2 or any(not isinstance(s, str) or not s.strip() for s in options):
                raise ValueError(f"sample_id={sid}: AH needs exactly two nonempty correction options")
        elif task == "dh":
            if not isinstance(sample.get("c"), str) or not sample["c"].strip():
                raise ValueError(f"sample_id={sid}: DH needs a nonempty correction")
        else:
            for _, _, field in CLIP_GROUPS:
                indices = sample.get(field)
                if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 1 for i in indices):
                    raise ValueError(f"sample_id={sid}: {field} needs positive 1-based frame indices")
    return samples


def build_prompt(task, sample):
    question = sample["q"].strip()
    constraint = "Do not output any other words, punctuation, or explanation.\n"
    if task == "dh":
        return (
            f"{question}\n"
            f"Correction: {sample['c'].strip()}\n"
            "A. Yes\n"
            "B. No\n"
            "Answer with exactly one letter. " + constraint
        )
    if task in ("ah1", "ah2"):
        return (
            f"{question}\n"
            f"A. {sample['c'][0].strip()}\n"
            f"B. {sample['c'][1].strip()}\n"
            "Answer with exactly one letter. " + constraint
        )
    return (
        f"{question}\n"
        "A. Candidate A\n"
        "B. Candidate B\n"
        "C. Candidate C\n"
        "Answer with exactly one letter (A/B/C). " + constraint
    )


def normalize_answer(raw, task):
    """Accept supported short forms or a unique option/choice reference.

    Unlike the old substring fallback, contradictory A/B/C references never
    silently become a valid answer. Invalid answers are eligible for retry.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    short = re.fullmatch(r"(?:(?:option|choice)\s+)?([abc])[.!?]?", text, re.I)
    if short:
        answer = short[1].upper()
        return answer if answer in choices(task) else None
    referenced = set(re.findall(r"\b(?:option|choice)\s+([abc])\b", text, re.I))
    mentioned = set(s.upper() for s in re.findall(r"\b([abc])\b", text, re.I))
    if referenced and len(mentioned) == 1:
        answer = next(iter(mentioned))
        return answer if answer in choices(task) else None
    return None
