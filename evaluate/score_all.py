import json
import argparse
from typing import Dict, Tuple, Any, Optional

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object at top level")
    return data

def has_valid_path(path: str) -> bool:
    return bool(path and str(path).strip())

def normalize_ab_choice(output_text: str) -> str:
    text = (output_text or "").strip()
    lower = text.lower()
    if lower in {"a", "a.", "a!", "a?", "option a", "choice a"}:
        return "A"
    if lower in {"b", "b.", "b!", "b?", "option b", "choice b"}:
        return "B"

    has_a = ("option a" in lower) or ("choice a" in lower)
    has_b = ("option b" in lower) or ("choice b" in lower)
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return text

def normalize_abc_choice(output_text: str) -> str:
    text = (output_text or "").strip()
    lower = text.lower()
    if lower in {"a", "a.", "a!", "a?", "option a", "choice a"}:
        return "A"
    if lower in {"b", "b.", "b!", "b?", "option b", "choice b"}:
        return "B"
    if lower in {"c", "c.", "c!", "c?", "option c", "choice c"}:
        return "C"

    has_a = ("option a" in lower) or ("choice a" in lower)
    has_b = ("option b" in lower) or ("choice b" in lower)
    has_c = ("option c" in lower) or ("choice c" in lower)
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    if has_c and not (has_a or has_b):
        return "C"
    return text

def score_mcq(predicted_answer, correct_answer):
    """Compute overall accuracy for A/B questions."""
    if len(predicted_answer) != len(correct_answer):
        raise ValueError(f"Length mismatch: predicted_answer has length {len(predicted_answer)}, but correct_answer has length {len(correct_answer)}")
    total = len(predicted_answer)
    correct = 0
    for id, ans in correct_answer.items():
        if id not in predicted_answer:
            raise ValueError(f"Missing prediction for id: {id}")
        if normalize_ab_choice(predicted_answer[id]) == normalize_ab_choice(ans['a']):
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def score_binary(predicted_answer, correct_answer):
    """Compute accuracy for odd/even ids and overall (A/B)."""
    if len(predicted_answer) != len(correct_answer):
        raise ValueError(f"Length mismatch: predicted_answer has length {len(predicted_answer)}, but correct_answer has length {len(correct_answer)}")
    total_odd = 0
    correct_odd = 0
    total_even = 0
    correct_even = 0
    for id, ans in correct_answer.items():
        if id not in predicted_answer:
            raise ValueError(f"Missing prediction for id: {id}")
        if int(id) % 2 == 1:  # Odd ID
            total_odd += 1
            if normalize_ab_choice(predicted_answer[id]) == normalize_ab_choice(ans['a']):
                correct_odd += 1
        else:  # Even ID
            total_even += 1
            if normalize_ab_choice(predicted_answer[id]) == normalize_ab_choice(ans['a']):
                correct_even += 1
    accuracy_odd = correct_odd / total_odd if total_odd > 0 else 0.0
    accuracy_even = correct_even / total_even if total_even > 0 else 0.0
    accuracy_average = (correct_odd + correct_even) / (total_odd + total_even) if (total_odd + total_even) > 0 else 0.0
    
    return accuracy_odd, accuracy_even, accuracy_average

def score_clip(predicted_answer, correct_answer):
    """Compute overall accuracy for A/B/C questions."""
    if len(predicted_answer) != len(correct_answer):
        raise ValueError(f"Length mismatch: predicted_answer has length {len(predicted_answer)}, but correct_answer has length {len(correct_answer)}")
    total = len(predicted_answer)
    correct = 0
    for id, ans in correct_answer.items():
        if id not in predicted_answer:
            raise ValueError(f"Missing prediction for id: {id}")
        if normalize_abc_choice(predicted_answer[id]) == normalize_abc_choice(ans['a']):
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score DH/TH/AH predictions.")
    parser.add_argument("--dh-gt", required=True, help="Ground truth JSON for DH (BinaryQA).")
    parser.add_argument("--th-gt", required=True, help="Ground truth JSON for TH (ClipQA).")
    parser.add_argument("--ah1-gt", required=True, help="Ground truth JSON for AH_1 (MCQ_1).")
    parser.add_argument("--ah2-gt", required=True, help="Ground truth JSON for AH_2 (MCQ_2).")

    parser.add_argument("--dh-pred", default="", help="Predictions JSON for DH.")
    parser.add_argument("--th-pred", default="", help="Predictions JSON for TH.")
    parser.add_argument("--ah1-pred", default="", help="Predictions JSON for AH_1.")
    parser.add_argument("--ah2-pred", default="", help="Predictions JSON for AH_2.")

    parser.add_argument("--out", default="", help="Output path to write scores JSON (optional).")
    return parser.parse_args()


def compute_scores(
    *,
    dh_pred_path: str,
    th_pred_path: str,
    ah1_pred_path: str,
    ah2_pred_path: str,
    dh_gt_path: str,
    th_gt_path: str,
    ah1_gt_path: str,
    ah2_gt_path: str,
) -> Dict[str, Any]:
    dh_answer = load_json(dh_gt_path)
    th_answer = load_json(th_gt_path)
    ah1_answer = load_json(ah1_gt_path)
    ah2_answer = load_json(ah2_gt_path)

    if has_valid_path(dh_pred_path):
        dh_pred = load_json(dh_pred_path)
        dh_accuracy_odd, dh_accuracy_even, dh_accuracy_average = score_binary(dh_pred, dh_answer)
    else:
        dh_accuracy_odd, dh_accuracy_even, dh_accuracy_average = 0.0, 0.0, 0.0

    if has_valid_path(th_pred_path):
        th_pred = load_json(th_pred_path)
        th_accuracy_average, _, _ = score_clip(th_pred, th_answer)
    else:
        th_accuracy_average = 0.0

    if has_valid_path(ah1_pred_path):
        ah1_pred = load_json(ah1_pred_path)
        ah1_accuracy, ah1_correct, ah1_total = score_mcq(ah1_pred, ah1_answer)
    else:
        ah1_accuracy = 0.0
        ah1_correct = 0
        ah1_total = len(ah1_answer)

    if has_valid_path(ah2_pred_path):
        ah2_pred = load_json(ah2_pred_path)
        ah2_accuracy, ah2_correct, ah2_total = score_mcq(ah2_pred, ah2_answer)
    else:
        ah2_accuracy = 0.0
        ah2_correct = 0
        ah2_total = len(ah2_answer)

    ah_total = ah1_total + ah2_total
    ah_accuracy_average = (ah1_correct + ah2_correct) / ah_total if ah_total > 0 else 0.0

    overall_accuracy = (dh_accuracy_average + th_accuracy_average + ah_accuracy_average) / 3
    return {
        "DH": {
            "accuracy_odd": dh_accuracy_odd,
            "accuracy_even": dh_accuracy_even,
            "accuracy_average": dh_accuracy_average,
        },
        "TH": {
            "accuracy_average": th_accuracy_average,
        },
        "AH_1": {
            "accuracy": ah1_accuracy,
        },
        "AH_2": {
            "accuracy": ah2_accuracy,
        },
        "AH": {
            "accuracy_average": ah_accuracy_average,
        },
        "overall": {
            "accuracy": overall_accuracy,
        },
    }


def main() -> None:
    args = parse_args()
    result = compute_scores(
        dh_pred_path=args.dh_pred,
        th_pred_path=args.th_pred,
        ah1_pred_path=args.ah1_pred,
        ah2_pred_path=args.ah2_pred,
        dh_gt_path=args.dh_gt,
        th_gt_path=args.th_gt,
        ah1_gt_path=args.ah1_gt,
        ah2_gt_path=args.ah2_gt,
    )

    if has_valid_path(args.out):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(args.out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    
if __name__ == "__main__":
    main()
