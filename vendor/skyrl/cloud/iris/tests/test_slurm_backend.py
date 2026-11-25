"""Behavior tests for the Slurm launch backend, run against fake Slurm commands."""

import os
import signal
import stat
import tarfile
import threading
import time
from pathlib import Path

import pytest
import yaml

from cloud.iris import launch
from cloud.iris.launch import LaunchOutcome, LaunchState, execute_launch
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.slurm.slurm_backend import SlurmBackend, render_job

JOB_ID = "4242"
SECRET = "s3cr3t value with spaces"

# The fake sbatch runs the batch script in the foreground, the fake srun runs the node script as rank 0 of
# one task, and the fake interpreter records every invocation, so the whole chain from launch to node program runs.
FAKE_COMMANDS = {
    "sbatch": f"""#!/usr/bin/env bash
script="${{@: -1}}"
echo "Welcome to a site whose submit hook prints a banner"
echo "--> Verifying valid jobname...OK"
echo "{JOB_ID}"
touch "$FAKE_SLURM_DIR/submitted"
[ "$(cat "$FAKE_SLURM_DIR/mode")" = hang ] && exit 0
[ -e "$FAKE_SLURM_DIR/vanishing_file" ] && rm -f "$(cat "$FAKE_SLURM_DIR/vanishing_file")"
export SLURM_JOB_ID={JOB_ID} SLURM_JOB_NODELIST=node1 SLURM_JOB_NUM_NODES=1
bash "$script" > "$FAKE_SLURM_DIR/job.out" 2>&1
status=$?
echo "$status" > "$FAKE_SLURM_DIR/code"
if [ "$status" -eq 0 ]; then echo COMPLETED > "$FAKE_SLURM_DIR/state"; else echo FAILED > "$FAKE_SLURM_DIR/state"; fi
""",
    "sbcast": """#!/usr/bin/env bash
echo "$*" >> "$FAKE_SLURM_DIR/sbcast.log"
cp -f "${@: -2:1}" "${@: -1}"
""",
    "srun": """#!/usr/bin/env bash
SLURM_PROCID=0 SLURM_NTASKS=1 exec bash "${@: -1}"
""",
    "squeue": """#!/usr/bin/env bash
[ -e "$FAKE_SLURM_DIR/stuck_completing" ] && { echo COMPLETING; exit 0; }
[ "$(cat "$FAKE_SLURM_DIR/mode")" = hang ] && echo RUNNING
exit 0
""",
    "sacct": """#!/usr/bin/env bash
echo "$(cat "$FAKE_SLURM_DIR/state")|$(cat "$FAKE_SLURM_DIR/code"):0|2026-11-21T10:00:00|2026-11-21T10:05:00"
""",
    "scancel": """#!/usr/bin/env bash
echo "$@" >> "$FAKE_SLURM_DIR/cancelled"
""",
    "python": """#!/usr/bin/env bash
dir="$FAKE_SLURM_DIR/invocations"; mkdir -p "$dir"
n=$(ls "$dir"/*.args 2>/dev/null | wc -l)
echo "$*" > "$dir/$n.args"
env > "$dir/$n.env"
echo "$0" > "$dir/$n.exe"
ls "$(dirname "$0")/../lib64" > "$dir/$n.lib" 2>/dev/null
head -1 "$(dirname "$0")/ray" > "$dir/$n.ray" 2>/dev/null
[ -n "$FAKE_PYTHON_EXIT" ] && exit "$FAKE_PYTHON_EXIT"
exit 0
""",
}


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    slurm_dir = tmp_path / "fake-slurm"
    bin_dir = slurm_dir / "bin"
    bin_dir.mkdir(parents=True)
    for name, text in FAKE_COMMANDS.items():
        if name != "python":  # the fake interpreter lives only in the job's runtime environment
            _executable(bin_dir / name, text)
    (slurm_dir / "mode").write_text("run")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(slurm_dir))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return slurm_dir


def _config_path(
    tmp_path: Path,
    *,
    submission: str = "wait",
    export_hf: bool = False,
    staged: bool = False,
    slurm_extra: dict | None = None,
) -> Path:
    runtime_env = tmp_path / "runtime-env"
    (runtime_env / "bin").mkdir(parents=True)
    _executable(runtime_env / "bin" / "python", FAKE_COMMANDS["python"])
    _executable(runtime_env / "bin" / "ray", f"#!{runtime_env}/bin/python\nprint('ray')\n")
    (runtime_env / "pyvenv.cfg").write_text("home = /usr/bin\n")
    activation = tmp_path / "runtime.env"
    activation.write_text(f"export FAKE_ACTIVATED=1\nexport FAKE_ENV_ROOT={runtime_env}\n")
    config = raw_launch_config()
    config["run"]["submission"] = submission
    config["run"]["export_hf"] = export_hf
    config["ray"].update(rendezvous_dir=str(tmp_path / "rendezvous"), log_dir=str(tmp_path / "ray-logs"))
    config["artifacts"].update(
        checkpoint_root=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        attempts_root=str(tmp_path / "attempts"),
        resolved_config_uri=str(tmp_path / "resolved.yaml"),
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
    )
    config["inputs"]["train_data"] = [
        {
            "uri": "s3://data/gsm8k",
            "identity": "sha256:gsm8k",
            "local_path": "/tmp/data/gsm8k",
            "relative_path": "train.parquet",
            "kind": "directory",
        }
    ]
    if not export_hf:  # HF export is an Iris job and needs the Iris-only settings
        for iris_only in ("cluster", "cluster_config"):
            del config["iris"][iris_only]
        for iris_only in ("cpu", "memory", "disk"):
            del config["iris"]["allocation"][iris_only]
    config["slurm"] = {
        "run_dir": str(tmp_path / "slurm-runs"),
        "time_limit": "01:00:00",
        "runtime_env": str(runtime_env),
        "activation_file": str(activation),
        "partition": "debug",
        "forward_env": ["FAKE_SECRET"],
        # The node program starts from a clean environment, so the fake interpreter's own variables are passed on.
        "env": {"NCCL_DEBUG": "WARN", **{name: os.environ[name] for name in ("FAKE_SLURM_DIR", "FAKE_PYTHON_EXIT") if name in os.environ}},
        "poll_interval": 1,
    }
    if staged:
        site_packages = runtime_env / "lib" / "python3.12" / "site-packages"
        site_packages.mkdir(parents=True)
        (site_packages / "libcudart.so.13").write_text("runtime")
        # The bootstrap links the unversioned linker name beside site-packages; the archive must keep it.
        (runtime_env / "lib" / "libcudart.so").symlink_to(site_packages / "libcudart.so.13")
        archive = tmp_path / "runtime-lib.tar"
        with tarfile.open(archive, "w", dereference=True) as tar:
            tar.add(runtime_env / "lib", arcname="lib")
        (tmp_path / "node-local").mkdir()
        config["slurm"].update(runtime_archive=str(archive), node_local_dir=str(tmp_path / "node-local"))
    config["slurm"].update(slurm_extra or {})
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _terminate_once_submitted(slurm_dir: Path) -> None:
    def signal_launcher() -> None:
        while not (slurm_dir / "submitted").exists():
            time.sleep(0.05)
        time.sleep(0.5)  # let the launcher reach its wait loop
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=signal_launcher, daemon=True).start()


def _invocation_env(slurm_dir: Path, index: int) -> dict[str, str]:
    lines = (slurm_dir / "invocations" / f"{index}.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def test_launch_runs_the_node_program_with_slurm_ranks_and_forwarded_secret(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)

    outcome = SlurmBackend().launch(_config_path(tmp_path))

    assert outcome == LaunchOutcome(JOB_ID, "succeeded", 0, "2026-11-21T10:00:00", "2026-11-21T10:05:00")
    assert len(list((fake_slurm / "invocations").glob("*.args"))) == 1  # the runtime
    runtime_env = _invocation_env(fake_slurm, 0)
    assert runtime_env["FAKE_ACTIVATED"] == "1"
    assert runtime_env["IRIS_TASK_ID"] == "0"
    assert runtime_env["IRIS_NUM_TASKS"] == "1"
    assert runtime_env["IRIS_PORT_RAY_HEAD"] == "0"
    assert runtime_env["NCCL_DEBUG"] == "WARN"
    assert runtime_env["FAKE_SECRET"] == SECRET
    launched_config = Path((fake_slurm / "invocations" / "0.args").read_text().split("--config ")[1].strip())
    assert yaml.safe_load(launched_config.read_text())["run"]["id"] == "smoke"


def test_launch_never_writes_the_secret_to_disk(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    SlurmBackend().launch(_config_path(tmp_path))

    rendered = [path for path in (tmp_path / "slurm-runs").rglob("*") if path.is_file()]
    assert rendered
    assert all(SECRET not in path.read_text() for path in rendered)


def test_failed_node_program_reports_a_failed_job(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    monkeypatch.setenv("FAKE_PYTHON_EXIT", "3")

    outcome = SlurmBackend().launch(_config_path(tmp_path))

    assert outcome.job_state == "failed"
    assert outcome.exit_code != 0


def test_missing_secret_blocks_submission(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE_SECRET", raising=False)

    with pytest.raises(ValueError, match="FAKE_SECRET"):
        SlurmBackend().launch(_config_path(tmp_path))

    assert not (fake_slurm / "state").exists()


def test_interrupting_the_launcher_cancels_the_job(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    (fake_slurm / "mode").write_text("hang")
    _terminate_once_submitted(fake_slurm)

    with pytest.raises(SystemExit) as interrupted:
        SlurmBackend().launch(_config_path(tmp_path))

    assert interrupted.value.code == 128 + signal.SIGTERM
    assert (fake_slurm / "cancelled").read_text().split() == [JOB_ID]


def test_detached_launch_returns_after_submission(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    (fake_slurm / "mode").write_text("hang")

    outcome = SlurmBackend().launch(_config_path(tmp_path, submission="detach"))

    assert outcome == LaunchOutcome(JOB_ID, "submitted", 0)


def test_hf_export_request_is_rejected_before_submission(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Export would fail only after training finished, so it must fail before the job is submitted.
    monkeypatch.setenv("FAKE_SECRET", SECRET)

    with pytest.raises(ValueError, match="export_hf"):
        SlurmBackend().launch(_config_path(tmp_path, export_hf=True))

    assert not (fake_slurm / "submitted").exists()


def test_staged_runtime_runs_from_the_node_local_copy_and_is_removed_afterwards(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)

    outcome = SlurmBackend().launch(_config_path(tmp_path, staged=True))

    node_local = tmp_path / "node-local"
    assert outcome.job_state == "succeeded"
    assert Path((fake_slurm / "invocations" / "0.exe").read_text().strip()).is_relative_to(node_local)
    assert Path(_invocation_env(fake_slurm, 0)["FAKE_ENV_ROOT"]).is_relative_to(node_local)
    assert (fake_slurm / "invocations" / "0.ray").read_text().startswith(f"#!{node_local}")
    assert "libcudart.so" in (fake_slurm / "invocations" / "0.lib").read_text().split()  # what JIT linking needs
    assert list(node_local.iterdir()) == []


def test_broadcast_sends_the_runtime_archive_to_the_nodes_once_by_the_batch_script(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    monkeypatch.setenv("USER", "tester")

    outcome = SlurmBackend().launch(
        _config_path(tmp_path, staged=True, slurm_extra={"broadcast_runtime_archive": True})
    )

    node_local = tmp_path / "node-local"
    assert outcome.job_state == "succeeded"
    sent = (fake_slurm / "sbcast.log").read_text().splitlines()
    assert len(sent) == 1 and sent[0].split()[-2:] == [str(tmp_path / "runtime-lib.tar"), f"{node_local}/skyrl-tester-archive.tar"]
    assert Path((fake_slurm / "invocations" / "0.exe").read_text().strip()).is_relative_to(node_local)
    assert list(node_local.iterdir()) == []  # the staged copy and the broadcast archive are both removed


def test_runtime_archive_is_read_directly_from_the_shared_filesystem_by_default(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    monkeypatch.setenv("USER", "tester")

    outcome = SlurmBackend().launch(_config_path(tmp_path, staged=True))

    node_local = tmp_path / "node-local"
    assert outcome.job_state == "succeeded"
    assert not (fake_slurm / "sbcast.log").exists()
    assert Path((fake_slurm / "invocations" / "0.exe").read_text().strip()).is_relative_to(node_local)  # still staged on the node
    assert list(node_local.iterdir()) == []


def test_node_that_cannot_read_a_preflight_file_fails_the_job_and_names_itself(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    (fake_slurm / "vanishing_file").write_text(str(weights))  # gone by the time the node reads it

    outcome = SlurmBackend().launch(_config_path(tmp_path, slurm_extra={"preflight_files": [str(weights)]}))

    assert outcome.job_state == "failed"
    log = (fake_slurm / "job.out").read_text()
    assert f"could not read" in log and str(weights) in log
    assert not (fake_slurm / "invocations").exists()  # the node program never started


def test_readable_preflight_files_do_not_block_the_job(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")

    outcome = SlurmBackend().launch(_config_path(tmp_path, slurm_extra={"preflight_files": [str(weights)]}))

    assert outcome.job_state == "succeeded"


def test_missing_preflight_file_is_rejected_before_submission(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", SECRET)

    with pytest.raises(FileNotFoundError, match="preflight"):
        SlurmBackend().launch(_config_path(tmp_path, slurm_extra={"preflight_files": [str(tmp_path / "absent")]}))

    assert not (fake_slurm / "submitted").exists()


def test_node_program_starts_from_a_clean_environment(tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Site modules export CC and similar variables that break Triton's compile step on the node.
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    monkeypatch.setenv("CC", "/site/module/bin/nvc")
    monkeypatch.setenv("LEAKED_FROM_LOGIN_SHELL", "1")

    SlurmBackend().launch(_config_path(tmp_path))

    environment = _invocation_env(fake_slurm, 0)
    assert "CC" not in environment
    assert "LEAKED_FROM_LOGIN_SHELL" not in environment
    assert environment["FAKE_SECRET"] == SECRET
    assert environment["SLURM_PROCID"] == "0"


def test_job_stuck_in_completing_still_reports_its_outcome(
    tmp_path: Path, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Horizon keeps a finished job COMPLETING for minutes while its nodes clean up; the launcher must not wait for that.
    monkeypatch.setenv("FAKE_SECRET", SECRET)
    (fake_slurm / "stuck_completing").touch()

    outcome = SlurmBackend().launch(_config_path(tmp_path))

    assert outcome.job_state == "succeeded"


class _Selected(Exception):
    """Raised by a recording backend as soon as the launcher validates with it, so no job lifecycle runs."""


def _recording_backend(name: str) -> type:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def validate(self, config_path: Path) -> None:
            raise _Selected(name)

    return Recorder


def _without_slurm_section(path: Path) -> Path:
    config = yaml.safe_load(path.read_text())
    del config["slurm"]
    stripped = path.with_name("iris-only.yaml")
    stripped.write_text(yaml.safe_dump(config, sort_keys=False))
    return stripped


def test_slurm_launch_command_runs_the_slurm_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cloud.slurm.slurm_backend.SlurmBackend", _recording_backend("slurm"))

    with pytest.raises(_Selected, match="slurm"):
        execute_launch(_config_path(tmp_path, submission="prepare"), component="slurm")


def test_iris_launch_command_runs_the_iris_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cloud.iris.iris_backend.IrisBackend", _recording_backend("iris"))

    with pytest.raises(_Selected, match="iris"):
        execute_launch(_without_slurm_section(_config_path(tmp_path, submission="prepare")))


def test_slurm_launch_without_a_slurm_section_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slurm: section"):
        execute_launch(_without_slurm_section(_config_path(tmp_path)), component="slurm")


def test_iris_launch_of_a_slurm_config_points_to_the_slurm_command(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slurm launch"):
        execute_launch(_config_path(tmp_path))


def test_command_line_passes_the_component_to_the_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    components = []

    def fake_execute_launch(config_path: Path, *, component: str, forward_env: tuple[str, ...]) -> object:
        components.append(component)
        return type("Result", (), {"state": LaunchState.SUCCEEDED, "model_dump": lambda self, **_: {}})()

    monkeypatch.setattr(launch, "execute_launch", fake_execute_launch)

    for component in ("iris", "slurm"):
        assert launch.main([component, "launch", "--config", str(tmp_path / "launch.yaml")]) == 0

    assert components == ["iris", "slurm"]


EXAMPLE = Path(__file__).resolve().parents[2] / "slurm" / "examples" / "qwen_smoke.yaml"


def test_documented_example_config_validates_and_renders_a_job(tmp_path: Path) -> None:
    # The example in the docs is what a new user starts from, so it must pass the same checks as a real launch.
    root = tmp_path / "shared"
    config_path = tmp_path / "launch.yaml"
    config_path.write_text(EXAMPLE.read_text().replace("/shared", str(root)))
    (root / "envs" / "skyrl-runtime" / "bin").mkdir(parents=True)
    _executable(root / "envs" / "skyrl-runtime" / "bin" / "python", "#!/bin/sh\n")
    (root / "envs" / "skyrl-runtime.runtime.env").write_text("")
    (root / "envs" / "skyrl-runtime-site-packages.tar").write_bytes(b"")
    (root / "models" / "Qwen3-0.6B").mkdir(parents=True)
    (root / "models" / "Qwen3-0.6B" / "model.safetensors").write_bytes(b"")

    backend = SlurmBackend()
    backend.validate(config_path)

    prepared = backend._prepare(config_path)
    _node_script, sbatch_script = render_job(prepared, root / "runs" / "smoke" / "slurm" / "attempt-001")
    assert "#SBATCH --nodes=1" in sbatch_script and "#SBATCH --gpus-per-node=4" in sbatch_script
    assert "sbcast" not in sbatch_script  # each node extracts the archive from the shared filesystem
