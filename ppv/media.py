"""Decode and sample each complete video independently of its motion array."""

import base64
import hashlib
import math

from ppv.results import file_hash


LABELS = ("[VIDEO 1: USER PERFORMANCE]", "[VIDEO 2: REFERENCE PERFORMANCE]")


def sample_video(path, fps, frame_size):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(native_fps) or native_fps <= 0:
            raise ValueError(f"Invalid video FPS: {path}")
        interval = max(int(round(native_fps / fps)), 1)
        images, manifest = [], []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % interval == 0:
                frame = cv2.resize(frame, tuple(frame_size), interpolation=cv2.INTER_AREA)
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError(f"Cannot encode frame {index}: {path}")
                jpeg = buffer.tobytes()
                images.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}})
                manifest.append({"video_frame_0based": index, "time_seconds": index / native_fps,
                                 "jpeg_bytes": len(jpeg), "jpeg_sha256": hashlib.sha256(jpeg).hexdigest()})
            index += 1
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not images:
            raise ValueError(f"Video has no decodable frames: {path}")
        # Detect a premature decoder failure when the container supplies a frame count.
        if math.isfinite(reported) and reported > 0 and index < int(reported):
            raise ValueError(f"Decoded only {index} of {int(reported)} video frames: {path}")
        return images, {"path": str(path), "sha256": file_hash(path), "decoded_frames": index,
                        "native_fps": native_fps, "frame_interval": interval,
                        "sampled_frames": len(images), "frames": manifest}
    finally:
        capture.release()


def build_content(videos, prompt):
    content = []
    for label, images in zip(LABELS, videos):
        content.append({"type": "text", "text": label})
        content.extend(images)
    content.append({"type": "text", "text": prompt})
    return content
