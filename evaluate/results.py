"""Atomic predictions, experiment identity, and credential-safe attempt logs."""

import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    root = Path(__file__).resolve().parent
    # Scoring and tests do not affect the evaluated model's inputs or answers.
    names = ("run.py", "api_client.py", "tasks.py", "media.py", "results.py")
    return fingerprint({name: file_hash(root / name) for name in names})


def redact(text, secret=""):
    text = str(text)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=\r\n]+", "[IMAGE OMITTED]", text)


def sanitize(value, secret=""):
    if isinstance(value, str):
        return redact(value, secret)
    if isinstance(value, dict):
        return {redact(k, secret): sanitize(v, secret) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secret) for v in value]
    return value


class _Formatter(logging.Formatter):
    def __init__(self, secret):
        super().__init__("%(asctime)s %(levelname)s %(message)s")
        self.secret = secret

    def format(self, record):
        return redact(super().format(record), self.secret)


@contextmanager
def task_logger(directory, secret):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger("motionhalluc.evaluate", level=logging.INFO)
    handlers = [logging.FileHandler(directory / "run.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(_Formatter(secret))
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


class ResultStore:
    def __init__(self, directory, config, secret=""):
        self.directory = Path(directory)
        self.secret = secret
        self.predictions_path = self.directory / "predictions.json"
        config = sanitize(config, secret)
        path = self.directory / "config.json"
        if path.exists():
            if read_json(path) != config:
                raise ValueError(f"Run configuration changed: choose a new --output-dir ({self.directory})")
        elif self.predictions_path.exists() or (self.directory / "details.jsonl").exists():
            raise ValueError(f"Existing results have no config.json: {self.directory}")
        else:
            write_json(path, config)
        self.predictions = read_json(self.predictions_path) if self.predictions_path.exists() else {}
        allowed = "ABC" if config["task"] == "th" else "AB"
        if not isinstance(self.predictions, dict) or any(
            sid not in config["sample_ids"] or not isinstance(answer, str) or len(answer) != 1 or answer not in allowed
            for sid, answer in self.predictions.items()
        ):
            raise ValueError(f"Existing predictions contain unexpected IDs or invalid answers: {self.directory}")

    def record(self, detail, answer=None):
        with (self.directory / "details.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sanitize(detail, self.secret), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if answer is not None:
            self.predictions[detail["sample_id"]] = answer
            write_json(self.predictions_path, self.predictions)
