from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def file_record(path: Path, role: str, *, hash_full: bool = True) -> dict:
    if not path.is_file():
        return {"path": str(path), "role": role, "status": "missing"}
    st = path.stat()
    rec: dict = {
        "path": str(path.resolve()),
        "role": role,
        "status": "present",
        "bytes": st.st_size,
    }
    if hash_full and st.st_size <= 50_000_000:
        rec["hash_kind"] = "sha256_full"
        rec["sha256"] = sha256_file(path)
    else:
        rec["hash_kind"] = "sha256_first_1mb_prefix"
        rec["sha256_prefix_1mb"] = sha256_file(path, max_bytes=1 << 20)
    return rec
