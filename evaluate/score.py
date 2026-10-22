"""Score saved base predictions independently, without API credentials or videos."""

import argparse
import sys
from pathlib import Path

from evaluate.results import file_hash, read_json, write_json
from evaluate.tasks import TASK_FILES, choices, load_samples, normalize_answer, sample_path, selected_tasks


def task_scores(task, samples_path, directory):
    directory = Path(directory)
    config = read_json(directory / "config.json")
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("task") != task or config.get("mode") != "base":
        raise ValueError(f"Invalid run configuration for {task}: {directory}")
    if config.get("samples_hash") != file_hash(samples_path):
        raise ValueError(f"Annotation fingerprint changed for {task}; provide the exact evaluated annotation file")
    samples = load_samples(samples_path, task)
    ids = config.get("sample_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(sid, str) for sid in ids) or len(set(ids)) != len(ids) or any(sid not in samples for sid in ids):
        raise ValueError(f"Invalid selected sample IDs for {task}")
    predictions = read_json(directory / "predictions.json")
    if not isinstance(predictions, dict) or set(predictions) != set(ids):
        raise ValueError(f"{task}: prediction IDs must exactly match selected IDs; finish/retry evaluation first")
    correct = 0
    for sid in ids:
        expected = samples[sid].get("a")
        if not isinstance(expected, str) or len(expected) != 1 or expected not in choices(task):
            raise ValueError(f"{task} sample_id={sid}: missing or invalid ground truth")
        answer = normalize_answer(predictions[sid], task)
        if answer is None:
            raise ValueError(f"{task} sample_id={sid}: invalid prediction")
        correct += answer == expected
    return {"correct": correct, "total": len(ids), "accuracy": correct / len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASK_FILES, "all"], default="all")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/MotionHalluc"))
    parser.add_argument("--samples-json", type=Path, help="Custom annotations; single-task scoring only")
    parser.add_argument("--output-dir", required=True, type=Path, help="Evaluation run root containing task directories")
    args = parser.parse_args(argv)
    if args.task == "all" and args.samples_json is not None:
        parser.error("--samples-json is only supported with a single --task")
    try:
        # Validate all requested results before writing any scores.
        scores = {task.upper(): task_scores(task, sample_path(task, args.dataset_dir, args.samples_json),
                                           args.output_dir / task) for task in selected_tasks(args.task)}
        for task in selected_tasks(args.task):
            write_json(args.output_dir / task / "scores.json", {task.upper(): scores[task.upper()]})
        if args.task == "all":
            write_json(args.output_dir / "scores.json", scores)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"Scoring stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"{'Task':<6} {'Correct':>8} {'Total':>8} {'Accuracy':>12}")
    for task, values in scores.items():
        print(f"{task:<6} {values['correct']:>8} {values['total']:>8} {values['accuracy']:>11.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
