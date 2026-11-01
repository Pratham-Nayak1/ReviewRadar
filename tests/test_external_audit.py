from pathlib import Path

from src.external_audit import md5


def test_external_artifact_checksum_is_reproducible(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"reviewradar")
    assert md5(artifact) == "c8fe0b18ed9387a1c9510f18f8ac000c"
