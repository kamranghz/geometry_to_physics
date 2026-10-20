from pathlib import Path

from geometry_to_physics.provenance.hashing import file_record


def test_hash_kind_full(tmp_path: Path):
    p = tmp_path / "small.txt"
    p.write_text("hello", encoding="utf-8")
    rec = file_record(p, "test")
    assert rec["hash_kind"] == "sha256_full"
    assert "sha256" in rec
    assert "sha256_prefix_1mb" not in rec


def test_hash_kind_prefix(tmp_path: Path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * (50_000_001))
    rec = file_record(p, "test")
    assert rec["hash_kind"] == "sha256_first_1mb_prefix"
    assert "sha256_prefix_1mb" in rec
    assert "sha256" not in rec
