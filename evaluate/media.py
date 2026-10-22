"""Full-video FPS sampling and exact TH annotation frames with reusable caches."""

import base64
import math
from pathlib import Path

from evaluate.results import fingerprint, read_json, write_json
from evaluate.tasks import CLIP_GROUPS, VIDEO_LABELS


def video_path(root, relative):
    path = Path(relative)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def video_identity(path):
    try:
        stat = Path(path).stat()
        return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except FileNotFoundError:
        return None


def extract_frames(path, size, cache_root, fps=None, indices=None):
    import cv2

    path = Path(path).resolve()
    identity = video_identity(path)
    if identity is None:
        raise FileNotFoundError(f"Video does not exist: {path}")
    key = fingerprint({"video": str(path), "identity": identity, "size": size,
                       "fps": fps, "indices": indices})
    directory = Path(cache_root) / key
    manifest = directory / "frames.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved and all((directory / item["file"]).is_file() for item in saved):
            return [(item["frame_1based"], directory / item["file"]) for item in saved]
    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video: {path}")
    frames = {}
    wanted = set(indices) if indices is not None else None
    try:
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(native_fps) or native_fps <= 0:
            native_fps = fps or 1
        interval = max(int(round(native_fps / (fps or 1))), 1)
        index = 0
        # Deliberately no frame-count cap: DH/AH read the entire video.
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            number = index + 1
            selected = number in wanted if wanted is not None else index % interval == 0
            if selected:
                file = directory / f"frame_{number:08d}.jpg"
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(file), frame):
                    raise RuntimeError(f"Failed to save frame: {file}")
                frames[number] = file
            index += 1
            if wanted is not None and wanted <= frames.keys():
                break
    finally:
        capture.release()
    if wanted is not None:
        missing = wanted - frames.keys()
        if missing:
            raise IndexError(f"Video {path.name} missing 1-based frames: {sorted(missing)}")
        selected = [(number, frames[number]) for number in indices]
    else:
        selected = list(frames.items())
    if not selected:
        raise RuntimeError(f"Video has no usable frames: {path}")
    write_json(manifest, [{"frame_1based": n, "file": p.name} for n, p in selected])
    return selected


def build_content(task, sample, prompt, video_root, cache_root, fps=4.0, size=(448, 448)):
    if task == "th":
        groups = [(label, video_path(video_root, sample[field]), sample[frames])
                  for label, field, frames in CLIP_GROUPS]
        # Decode each source once, then restore each candidate's original order.
        requested = {}
        for _, path, indices in groups:
            requested.setdefault(path, set()).update(indices)
        selected = {path: dict(extract_frames(path, size, cache_root, indices=sorted(indices)))
                    for path, indices in requested.items()}
    else:
        groups = [(label, video_path(video_root, sample[field]), None)
                  for label, field in zip(VIDEO_LABELS, ("v1", "v2"))]
        selected = {path: dict(extract_frames(path, size, cache_root, fps=fps))
                    for _, path, _ in groups}
    content, trace = [], []
    for label, path, indices in groups:
        frames = selected[path]
        ordered = indices if indices is not None else list(frames)
        content.append({"type": "text", "text": label})
        for number in ordered:
            encoded = base64.b64encode(frames[number].read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        trace.append({"label": label, "video": str(path), "frames_1based": ordered})
    content.append({"type": "text", "text": prompt})
    return content, trace
