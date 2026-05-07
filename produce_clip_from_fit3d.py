from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ClipJob:
	fit3d: Path
	bench_out: Path
	start_frame: int
	end_frame: int


FRAME_RE = re.compile(r"_(\d+)_(\d+)(?:\.[^.]+)?$")


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _parse_frame_range_from_bench_filename(bench_path: Path) -> Tuple[int, int]:
	"""Parse start/end frame indices from the output filename.

	The semantics are end-exclusive: [start, end).
	"""
	stem = bench_path.stem # filename without suffix
	m = FRAME_RE.search(stem)
	if not m:
		raise ValueError(
			f"Cannot parse frame range from output filename: {bench_path.name}. Expected '*_215_304.mp4'."
		)
	start = int(m.group(1))
	end = int(m.group(2))
	if start < 0 or end < 0 or end <= start:
		raise ValueError(f"Invalid frame range (end-exclusive): start={start}, end={end} ({bench_path.name})")
	return start, end


def _ensure_ffmpeg() -> str:
	exe = shutil.which("ffmpeg")
	if not exe:
		raise RuntimeError("ffmpeg not found. Please install it or ensure it is available in PATH.")
	return exe


def _run(cmd: List[str]) -> None:
	try:
		subprocess.run(cmd, check=True)
	except subprocess.CalledProcessError as exc:
		cmd_str = " ".join(shlex.quote(c) for c in cmd)
		raise RuntimeError(f"ffmpeg execution failed (exit={exc.returncode}): {cmd_str}") from exc


def _clip_by_frames(*, ffmpeg: str, input_mp4: Path, start: int, end: int, output_mp4: Path) -> None:
	"""Frame-accurate cut using ffmpeg's select filter.

	The input range is end-exclusive: [start, end).
	ffmpeg select(between) is inclusive, so we convert to [start, end-1].
	"""
	# n is 0-based frame index.
	end_inclusive = end - 1
	vf = f"select='between(n\\,{start}\\,{end_inclusive})',setpts=N/FRAME_RATE/TB"

	output_mp4.parent.mkdir(parents=True, exist_ok=True)
	cmd = [
		ffmpeg,
		"-hide_banner",
		"-loglevel",
		"error",
		"-y",
		"-i",
		str(input_mp4),
		"-vf",
		vf,
		"-vsync",
		"0",
		"-an",
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		"-movflags",
		"+faststart",
		str(output_mp4),
	]
	_run(cmd)


def _display_path(path: Path, *, root: Path) -> str:
	"""Return a sanitized path string for logs.

	- Prefer project-root-relative paths.
	- Never print absolute paths.
	- If not under root, fall back to filename only.
	"""
	try:
		return path.resolve().relative_to(root).as_posix()
	except Exception:
		return path.name


def _iter_jobs_from_pairs_file(pairs_path: Path, *, root: Path) -> Iterable[ClipJob]:
	for raw in pairs_path.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line or line.startswith("#"):
			continue

		parts = line.split()
		if len(parts) != 2:
			raise ValueError(f"Invalid line in pairs file (expected 2 columns): {raw}")

		fit3d = Path(parts[0])
		bench = Path(parts[1])
		if not fit3d.is_absolute():
			fit3d = root / fit3d
		if not bench.is_absolute():
			bench = root / bench

		start, end = _parse_frame_range_from_bench_filename(bench)
		yield ClipJob(fit3d=fit3d, bench_out=bench, start_frame=start, end_frame=end)


def _parse_args(argv: List[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Frame-accurate clipping from source mp4 to output mp4")
	parser.add_argument("--pairs", help="Two-column txt file (each line: source_mp4 output_mp4)")
	parser.add_argument("--fit3d", help="Single job: source mp4 path")
	parser.add_argument("--bench", help="Single job: output mp4 path (filename must include *_start_end.mp4)")
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting existing output files (ffmpeg uses -y by default)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print planned jobs only; do not run ffmpeg",
	)
	return parser.parse_args(argv)


def main(argv: List[str]) -> int:
	args = _parse_args(argv)
	root = _project_root()

	if not args.pairs and not (args.fit3d and args.bench):
		print("[ERR] Provide --pairs or both --fit3d and --bench", file=sys.stderr)
		return 2

	ffmpeg = _ensure_ffmpeg() if not args.dry_run else "ffmpeg"

	jobs: List[ClipJob] = []
	if args.pairs:
		pairs_path = Path(args.pairs)
		if not pairs_path.is_absolute():
			pairs_path = root / pairs_path
		if not pairs_path.exists():
			print(f"[ERR] Pairs file not found: {_display_path(pairs_path, root=root)}", file=sys.stderr)
			return 2
		jobs.extend(list(_iter_jobs_from_pairs_file(pairs_path, root=root)))
	else:
		fit3d = Path(args.fit3d)
		bench = Path(args.bench)
		if not fit3d.is_absolute():
			fit3d = root / fit3d
		if not bench.is_absolute():
			bench = root / bench
		start, end = _parse_frame_range_from_bench_filename(bench)
		jobs.append(ClipJob(fit3d=fit3d, bench_out=bench, start_frame=start, end_frame=end))

	if not jobs:
		print("[ERR] No jobs parsed", file=sys.stderr)
		return 3

	for job in jobs:
		if not job.fit3d.exists():
			print(
				f"[ERR] Source video not found: {_display_path(job.fit3d, root=root)}",
				file=sys.stderr,
			)
			return 4

		if job.bench_out.exists() and not args.overwrite:
			print(
				f"[ERR] Output exists and overwrite not allowed: {_display_path(job.bench_out, root=root)}",
				file=sys.stderr,
			)
			return 5

		count = job.end_frame - job.start_frame
		print(
			"clip: "
			+ _display_path(job.fit3d, root=root)
			+ " -> "
			+ _display_path(job.bench_out, root=root)
			+ f" (frames [{job.start_frame},{job.end_frame}) = {count})"
		)
		if args.dry_run:
			continue

		_clip_by_frames(
			ffmpeg=ffmpeg,
			input_mp4=job.fit3d,
			start=job.start_frame,
			end=job.end_frame,
			output_mp4=job.bench_out,
		)

	print(f"Done: processed {len(jobs)} clip(s)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))