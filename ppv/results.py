"""Secret-safe JSON event log and incremental atomic result snapshots."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunRecord:
    def __init__(self, root, secrets):
        self.secrets = sorted({key for key in secrets if key}, key=len, reverse=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(root) / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.result = {"status": "running", "started_at": self.now()}

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def clean(self, value):
        if isinstance(value, dict):
            return {self.clean(str(k)): "[REDACTED]" if str(k).lower() in
                    ("api_key", "authorization") else self.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
                for ascii_only in (False, True):
                    value = value.replace(json.dumps(secret, ensure_ascii=ascii_only)[1:-1], "[REDACTED]")
            return re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[IMAGE DATA OMITTED]", value)
        return value

    def log(self, event, **fields):
        row = self.clean({"time": self.now(), "event": event, **fields})
        with (self.directory / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def save(self, **fields):
        self.result.update(fields)
        self.result["updated_at"] = self.now()
        document = self.clean(self.result)
        fd, name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.directory / "result.json")
        finally:
            if os.path.exists(name):
                os.unlink(name)
