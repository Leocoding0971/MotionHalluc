import argparse
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import requests


def normalize_samples(raw_samples: Any) -> Dict[str, Dict[str, Any]]:
	"""Normalize samples into a dict keyed by string sample_id.

	Expected sample schema (per item):
	- sample_id: str|int (required if input is a list)
	- v1: str (query video path)
	- v2: str (candidate video path; A/B/C are different frame subsets from this video)
	- q: str (question)
	- f1: list[int] (1-based frame indices for v1)
	- f2_A/f2_B/f2_C: list[int] (1-based frame indices for v2 candidate A/B/C)
	"""
	if isinstance(raw_samples, dict):
		normalized: Dict[str, Dict[str, Any]] = {}
		for sample_id, sample in raw_samples.items():
			sid = str(sample_id)
			if not isinstance(sample, dict):
				raise ValueError(f"sample_id={sid} must map to an object")
			normalized[sid] = sample
		return normalized

	if isinstance(raw_samples, list):
		normalized = {}
		for item in raw_samples:
			if not isinstance(item, dict):
				raise ValueError("each list item must be an object")
			sid = str(item.get("sample_id", "")).strip()
			if not sid:
				raise ValueError("list item missing non-empty sample_id")
			normalized[sid] = item
		return normalized

	raise ValueError("samples must be a dict or a list")


def safe_join_video_path(video_root: str, relative_or_abs: str) -> str:
	if os.path.isabs(relative_or_abs):
		return relative_or_abs
	return os.path.join(video_root, relative_or_abs)


def cache_all_frames(
	video_path: str,
	frame_resolution: Tuple[int, int],
	cache_root: str,
	cache_namespace: str,
) -> List[str]:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError("failed to open video")

	video_hash = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
	video_stem = Path(video_path).stem
	temp_dir = Path(cache_root) / cache_namespace / f"{video_stem}_{video_hash}"
	done_marker = temp_dir / ".done"

	if temp_dir.exists() and temp_dir.is_dir() and done_marker.exists():
		cached_frames = sorted(
			[
				str(p)
				for p in temp_dir.iterdir()
				if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
			]
		)
		if cached_frames:
			cap.release()
			return cached_frames

	temp_dir.mkdir(parents=True, exist_ok=True)

	frame_paths: List[str] = []
	frame_idx = 0
	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break
		frame = cv2.resize(frame, frame_resolution, interpolation=cv2.INTER_AREA)
		out_path = temp_dir / f"frame_{frame_idx + 1:06d}.jpg"
		cv2.imwrite(str(out_path), frame)
		frame_paths.append(str(out_path))
		frame_idx += 1

	cap.release()
	done_marker.write_text("ok", encoding="utf-8")
	return frame_paths


def parse_required_frame_indices(sample: Dict[str, Any], field_name: str, sample_id: str) -> List[int]:
	if field_name not in sample:
		raise ValueError(f"sample_id={sample_id} missing required field: {field_name}")
	indices = sample[field_name]
	if not isinstance(indices, list) or len(indices) == 0:
		raise ValueError(f"sample_id={sample_id} field '{field_name}' must be a non-empty list")

	parsed: List[int] = []
	for idx in indices:
		if not isinstance(idx, int):
			raise ValueError(f"sample_id={sample_id} field '{field_name}' contains non-int index")
		if idx <= 0:
			raise ValueError(f"sample_id={sample_id} field '{field_name}' contains invalid 1-based index")
		parsed.append(idx)
	return parsed


def select_frames_by_1based_indices(
	all_frame_paths: List[str],
	frame_indices: List[int],
	sample_id: str,
	video_tag: str,
) -> List[str]:
	total_frames = len(all_frame_paths)
	selected: List[str] = []
	for idx in frame_indices:
		if idx > total_frames:
			raise IndexError(
				f"sample_id={sample_id} {video_tag} frame index out of range (total_frames={total_frames})"
			)
		selected.append(all_frame_paths[idx - 1])
	return selected


def encode_images_to_base64(image_paths: List[str]) -> List[str]:
	encoded: List[str] = []
	for image_path in image_paths:
		with open(image_path, "rb") as f:
			encoded.append(base64.b64encode(f.read()).decode("utf-8"))
	return encoded


def build_prompt(question: str) -> str:
	return (
		f"{question}\n"
		"A. Candidate A\n"
		"B. Candidate B\n"
		"C. Candidate C\n"
		"Answer with exactly one letter (A/B/C). "
		"Do not output any other words, punctuation, or explanation.\n"
	)


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
	if has_a and not has_b and not has_c:
		return "A"
	if has_b and not has_a and not has_c:
		return "B"
	if has_c and not has_a and not has_b:
		return "C"

	return text


def build_content(
	video1_frames_b64: List[str],
	video2_a_frames_b64: List[str],
	video2_b_frames_b64: List[str],
	video2_c_frames_b64: List[str],
	prompt: str,
) -> List[Dict[str, Any]]:
	content: List[Dict[str, Any]] = []
	content.append({"type": "text", "text": "[VIDEO 1]"})
	for b64 in video1_frames_b64:
		content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

	content.append({"type": "text", "text": "[CANDIDATE A]"})
	for b64 in video2_a_frames_b64:
		content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

	content.append({"type": "text", "text": "[CANDIDATE B]"})
	for b64 in video2_b_frames_b64:
		content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

	content.append({"type": "text", "text": "[CANDIDATE C]"})
	for b64 in video2_c_frames_b64:
		content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

	content.append({"type": "text", "text": prompt})
	return content


def call_chat_completions(
	*,
	api_url: str,
	api_key: str,
	model_name: str,
	content: List[Dict[str, Any]],
	max_tokens: int,
	request_timeout: int,
) -> str:
	payload = {
		"model": model_name,
		"messages": [{"role": "user", "content": content}],
		"max_tokens": max_tokens,
	}
	headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
	resp = requests.post(api_url, headers=headers, json=payload, timeout=request_timeout)
	if resp.status_code >= 400:
		raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
	data = resp.json()
	choices = data.get("choices") or []
	if not choices:
		raise RuntimeError("invalid response format: missing choices")
	message = choices[0].get("message") or {}
	content_out = message.get("content", "")
	if not isinstance(content_out, str):
		content_out = str(content_out)
	return content_out


def save_results_atomic(results: Dict[str, str], output_path: str) -> None:
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	fd, temp_path = tempfile.mkstemp(prefix="results_tmp_", suffix=".json", dir=str(Path(output_path).parent))
	with os.fdopen(fd, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)
		f.flush()
		os.fsync(f.fileno())
	os.replace(temp_path, output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Baseline TH evaluator (no parser/report).")
	parser.add_argument("--samples-json", required=True, help="Input samples json path.")
	parser.add_argument("--video-root", default=".", help="Base directory for relative video paths.")
	parser.add_argument("--output-json", default="th_results.json", help="Output json path.")
	parser.add_argument("--cache-root", default="./th_cache", help="Cache root directory.")
	parser.add_argument(
		"--cache-namespace",
		default=f"worker_{os.getpid()}",
		help="Per-process cache namespace under cache root.",
	)

	parser.add_argument("--api-url", required=True, help="Chat Completions API URL (OpenAI-compatible).")
	parser.add_argument(
		"--api-key",
		default=os.environ.get("API_KEY", ""),
		help="API key (or set env API_KEY).",
	)
	parser.add_argument("--model-name", required=True, help="Model name.")

	parser.add_argument("--frame-size", default="448,448", help="Frame size as 'W,H'.")
	parser.add_argument("--max-tokens", type=int, default=3000)
	parser.add_argument("--timeout", type=int, default=600)
	parser.add_argument(
		"--postprocess",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Normalize output to exactly 'A'/'B'/'C' when possible.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.api_key:
		raise SystemExit("Missing --api-key (or env API_KEY)")

	try:
		w_str, h_str = str(args.frame_size).split(",", 1)
		frame_resolution = (int(w_str), int(h_str))
	except Exception:
		raise SystemExit("Invalid --frame-size; expected 'W,H' (e.g. 448,448)")

	with open(args.samples_json, "r", encoding="utf-8") as f:
		samples_raw = json.load(f)
	samples = normalize_samples(samples_raw)

	results: Dict[str, str] = {}
	if os.path.exists(args.output_json):
		try:
			with open(args.output_json, "r", encoding="utf-8") as f:
				loaded = json.load(f)
				if isinstance(loaded, dict):
					results = loaded
		except Exception:
			results = {}

	done_ids = set(results.keys())
	api_failed_ids: List[str] = []

	for sample_id, sample in samples.items():
		sample_id = str(sample_id).strip()
		if not sample_id or sample_id in done_ids:
			continue
		if not isinstance(sample, dict):
			raise ValueError("each sample must be an object")

		for field in ("v1", "v2", "q", "f1", "f2_A", "f2_B", "f2_C"):
			if field not in sample:
				raise ValueError(f"sample_id={sample_id} missing required field: {field}")

		question = str(sample.get("q", "")).strip()
		if not question:
			raise ValueError(f"sample_id={sample_id} has empty question")

		f1_indices = parse_required_frame_indices(sample, "f1", sample_id)
		f2_a_indices = parse_required_frame_indices(sample, "f2_A", sample_id)
		f2_b_indices = parse_required_frame_indices(sample, "f2_B", sample_id)
		f2_c_indices = parse_required_frame_indices(sample, "f2_C", sample_id)

		prompt = build_prompt(question)

		video1_path = safe_join_video_path(args.video_root, str(sample["v1"]).strip())
		video2_path = safe_join_video_path(args.video_root, str(sample["v2"]).strip())

		video1_all_frames = cache_all_frames(video1_path, frame_resolution, args.cache_root, args.cache_namespace)
		video2_all_frames = cache_all_frames(video2_path, frame_resolution, args.cache_root, args.cache_namespace)

		video1_frames = select_frames_by_1based_indices(video1_all_frames, f1_indices, sample_id, "v1")
		video2_a_frames = select_frames_by_1based_indices(video2_all_frames, f2_a_indices, sample_id, "v2_A")
		video2_b_frames = select_frames_by_1based_indices(video2_all_frames, f2_b_indices, sample_id, "v2_B")
		video2_c_frames = select_frames_by_1based_indices(video2_all_frames, f2_c_indices, sample_id, "v2_C")

		video1_b64 = encode_images_to_base64(video1_frames)
		video2_a_b64 = encode_images_to_base64(video2_a_frames)
		video2_b_b64 = encode_images_to_base64(video2_b_frames)
		video2_c_b64 = encode_images_to_base64(video2_c_frames)
		content = build_content(video1_b64, video2_a_b64, video2_b_b64, video2_c_b64, prompt)

		try:
			raw_output = call_chat_completions(
				api_url=args.api_url,
				api_key=args.api_key,
				model_name=args.model_name,
				content=content,
				max_tokens=args.max_tokens,
				request_timeout=args.timeout,
			)
		except Exception as e:
			api_failed_ids.append(sample_id)
			raw_output = f"[ERROR] {type(e).__name__}: {e}"

		final_output = normalize_abc_choice(raw_output) if args.postprocess else raw_output
		results[sample_id] = final_output
		save_results_atomic(results, args.output_json)

	if api_failed_ids:
		print(f"API failures: {len(api_failed_ids)}")
	else:
		print("API failures: 0")


if __name__ == "__main__":
	main()
