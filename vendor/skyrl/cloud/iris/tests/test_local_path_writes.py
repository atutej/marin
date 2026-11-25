"""Writes to local paths must create their parent directories, as writes to an object store implicitly do."""

from pathlib import Path

from marinskyrl.remote_io import filesystem_and_path


def test_local_write_creates_missing_parent_directories(tmp_path: Path) -> None:
    # The Ray log and debug-artifact syncs write to rank-specific directories nobody created first.
    target = tmp_path / "rendezvous" / "debug_artifacts" / "rank0-node" / "sync-manifest.json"
    filesystem, path = filesystem_and_path(str(target))

    with filesystem.open(path, "w") as manifest:
        manifest.write("{}")

    assert target.read_text() == "{}"


def test_local_put_creates_missing_parent_directories(tmp_path: Path) -> None:
    source = tmp_path / "worker.err"
    source.write_text("log")
    destination = tmp_path / "ray-logs" / "rank0-node" / "session" / "worker.err"
    filesystem, path = filesystem_and_path(str(destination))

    filesystem.put(str(source), path)

    assert destination.read_text() == "log"
