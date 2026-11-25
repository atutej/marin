# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one resolved SkyRL launch config as a Slurm job and wait for its terminal state."""

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from iris.resources.state import JobState
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cloud.iris.launch import LaunchOutcome
from cloud.iris.launch_config import LaunchTopology, SubmissionMode, load_launch_config, validate_launch_config
from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.ray_ports import RayPort
from cloud.iris.ray_storage import RaySpillBackend, resolve_ray_spill_target
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT
from cloud.slurm.config import SlurmConfig
from cloud.slurm.scripts import (
    NodeScriptSpec,
    RuntimePaths,
    SbatchSpec,
    parse_job_id,
    render_node_script,
    render_runtime_paths,
    render_sbatch,
)

REPO_ROOT = PROJECT_ROOT.parents[1]
TASK_RUNTIME = "cloud/iris/task_runtime.py"
RAY_PORT_VARS = tuple(f"IRIS_PORT_{port.value.upper()}" for port in RayPort)
SKYRL_CONTROL_PREFIX = "SKYRL_"
ACTIVE_QUEUE_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
        "STOPPED",
    }
)
ACCOUNTING_RETRIES = 10
ACCOUNTING_RETRY_DELAY = 3
SUBMITTED_STATE = "submitted"
UNSET_TIMESTAMPS = frozenset({"Unknown", "None"})


@dataclass(frozen=True)
class PreparedLaunch:
    config: DictConfig
    settings: SlurmConfig
    topology: LaunchTopology


def slurm_settings(config: DictConfig) -> SlurmConfig:
    if config.get("slurm") is None:
        raise ValueError("The Slurm backend requires a slurm section in the launch config")
    settings = OmegaConf.to_object(config.slurm)
    assert isinstance(settings, SlurmConfig)
    return settings


def secret_names(config: DictConfig, settings: SlurmConfig, forward_env: tuple[str, ...]) -> tuple[str, ...]:
    """Environment variables the job needs from the submitting shell, in first-use order."""
    names = [*forward_env, *settings.forward_env]
    if config.runtime.entrypoint != CHECKPOINT_EXPORT_ENTRYPOINT:
        for backend in ("remote_judge", "daytona"):
            key_env = OmegaConf.select(config, f"skyrl.environment.task_sessions.{backend}.api_key_env")
            if key_env is not None:
                names.append(str(key_env))
        for callback in OmegaConf.select(config, "skyrl.trainer.callbacks", default=[]) or []:
            if callback.get("type") == "harbor_evaluation":
                names.append(str(callback["daytona_api_key_env"]))
    return tuple(dict.fromkeys(names))


def require_environment(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError(f"These environment variables are required by the job and unset here: {', '.join(missing)}")


def task_environment(config: DictConfig, settings: SlurmConfig) -> dict[str, str]:
    """Non-secret variables for every node: ``runtime.task_env`` then ``slurm.env``, plus the rendezvous directory."""
    controls = sorted(str(key) for key in config.runtime.task_env if str(key).startswith(SKYRL_CONTROL_PREFIX))
    if controls:
        raise ValueError(
            "RL config extra_env may not define MarinSkyRL behavior; use typed Hydra settings instead: "
            + ", ".join(controls)
        )
    environment = {
        str(key): str(int(value)) if isinstance(value, bool) else str(value)
        for key, value in config.runtime.task_env.items()
        if value is not None
    }
    environment.update(settings.env)
    environment["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = str(config.ray.rendezvous_dir)
    return environment


def node_body(config: DictConfig, settings: SlurmConfig, config_copy: Path) -> str:
    """The commands each node runs after its environment is set: runtime paths, spill check, activation, runtime.

    The Iris launcher also runs an interconnect gate here. It rendezvous through Iris's endpoint registry,
    so it cannot run under Slurm.
    """
    rendezvous_dir = str(config.ray.rendezvous_dir)
    spill_check = resolve_ray_spill_target(
        rendezvous_dir, RaySpillBackend(config.ray.spill_backend), str(config.ray.spill_dir)
    ).shell_preflight()
    paths = render_runtime_paths(
        RuntimePaths(
            runtime_env=settings.runtime_env,
            activation_file=settings.activation_file,
            repo_root=str(REPO_ROOT),
            archive=settings.runtime_archive,
            local_dir=settings.node_local_dir,
            broadcast=settings.runtime_archive is not None and settings.broadcast_runtime_archive,
        )
    )
    runtime = f'"$SKYRL_ENV/bin/python" {shlex.join([TASK_RUNTIME, "--config", str(config_copy)])}'
    skyrl_home = '"$SKYRL_REPO/vendor/skyrl"'
    return "\n".join(
        [
            spill_check,
            paths,
            f"cd {skyrl_home}",
            f"export SKYRL_HOME={skyrl_home}",
            'source "$SKYRL_ACTIVATION"',
            'export PYTHONPATH="$SKYRL_REPO:$SKYRL_REPO/vendor/skyrl:$SKYRL_REPO/vendor/skyrl/skyrl-train:${PYTHONPATH:-}"',
            "export VLLM_USE_V1=1",
            # Staged runtimes are removed by the shell's exit trap, which exec would discard.
            runtime if settings.runtime_archive is not None else f"exec {runtime}",
        ]
    )


def render_job(prepared: PreparedLaunch, run_dir: Path, forward_env: tuple[str, ...] = ()) -> tuple[str, str]:
    """Return the (node script, sbatch script) text for one launch rooted at ``run_dir``."""
    config, settings, topology = prepared.config, prepared.settings, prepared.topology
    node_script = render_node_script(
        NodeScriptSpec(
            forwarded_env=secret_names(config, settings, forward_env),
            env=task_environment(config, settings),
            ray_port_vars=RAY_PORT_VARS,
            advertise_interface=settings.advertise_interface,
            body=node_body(config, settings, run_dir / "launch.yaml"),
            preflight_files=tuple(settings.preflight_files),
        )
    )
    sbatch_script = render_sbatch(
        SbatchSpec(
            job_name=str(config.iris.job_name),
            nodes=topology.num_nodes,
            gpus_per_node=topology.gpus_per_node,
            time_limit=settings.time_limit,
            log_dir=str(run_dir / "logs"),
            node_script=str(run_dir / "node.sh"),
            partition=settings.partition,
            account=settings.account,
            qos=settings.qos,
            reservation=settings.reservation,
            constraint=settings.constraint,
            cpus_per_task=settings.cpus_per_task,
            extra_options=tuple(settings.extra_sbatch),
            broadcast_archive=settings.runtime_archive if settings.broadcast_runtime_archive else None,
            node_local_dir=settings.node_local_dir,
        )
    )
    return node_script, sbatch_script


def run_slurm_command(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@contextmanager
def cancel_on_signal(job_id: str) -> Iterator[None]:
    """Cancel the Slurm job when the launcher is interrupted, so a stopped launcher never leaves a job behind."""

    def cancel(signum: int, _frame: object) -> None:
        logger.warning(f"Received signal {signum}; cancelling Slurm job {job_id}")
        # Best effort: the job may already be gone, and the launcher is exiting either way.
        subprocess.run(["scancel", job_id], check=False)
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def job_state(accounting_state: str) -> JobState:
    state = accounting_state.split()[0]  # "CANCELLED by 1234" carries the canceller
    if state == "COMPLETED":
        return JobState.SUCCEEDED
    if state.startswith("CANCELLED"):
        return JobState.KILLED
    return JobState.FAILED


def accounting_row(job_id: str) -> list[str] | None:
    """The job's ``State|ExitCode|Start|End`` accounting fields, or None before the record exists."""
    rows = run_slurm_command(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State,ExitCode,Start,End"]).splitlines()
    return rows[0].split("|") if rows else None


def terminal_outcome(job_id: str) -> LaunchOutcome:
    for _ in range(ACCOUNTING_RETRIES):
        row = accounting_row(job_id)
        if row is not None:
            break
        time.sleep(ACCOUNTING_RETRY_DELAY)
    else:
        raise RuntimeError(f"Slurm accounting has no record of job {job_id}")
    accounting_state, exit_code, started_at, finished_at = row
    state = job_state(accounting_state)
    code = int(exit_code.split(":")[0])
    if state != JobState.SUCCEEDED and code == 0:
        code = 1
    return LaunchOutcome(
        job_id=job_id,
        job_state=state.value,
        exit_code=code,
        started_at=None if started_at in UNSET_TIMESTAMPS else started_at,
        finished_at=None if finished_at in UNSET_TIMESTAMPS else finished_at,
    )


def job_has_ended(job_id: str) -> bool:
    """Whether the queue has released the job, or its accounting record is final while the nodes still clean up."""
    queue_state = run_slurm_command(["squeue", "-h", "-j", job_id, "-o", "%T"]).strip()
    if queue_state not in ACTIVE_QUEUE_STATES:
        return True
    # A finished job stays COMPLETING until its nodes clean up, which can take minutes; its result is already known.
    row = accounting_row(job_id) if queue_state == "COMPLETING" else None
    return row is not None and row[0].split()[0] not in ACTIVE_QUEUE_STATES


def wait_for_job(job_id: str, poll_interval: int) -> LaunchOutcome:
    while not job_has_ended(job_id):
        time.sleep(poll_interval)
    return terminal_outcome(job_id)


class SlurmBackend:
    """Launch backend that submits a resolved config with ``sbatch`` and supervises the job."""

    def __init__(self, *, forward_env: tuple[str, ...] = ()):
        self._forward_env = forward_env

    def _prepare(self, config_path: Path) -> PreparedLaunch:
        config = load_launch_config(config_path)
        topology = validate_launch_config(config)
        settings = slurm_settings(config)
        if config.run.export_hf:
            raise ValueError("run.export_hf exports through an Iris job and is not supported on Slurm yet")
        if topology.role_topology is not None:
            raise ValueError("iris.role_topology relies on Iris coscheduling groups and is not supported on Slurm")
        if not Path(settings.runtime_env, "bin", "python").is_file():
            raise FileNotFoundError(f"slurm.runtime_env has no bin/python: {settings.runtime_env}")
        if not Path(settings.activation_file).is_file():
            raise FileNotFoundError(f"slurm.activation_file does not exist: {settings.activation_file}")
        if settings.runtime_archive is not None and not Path(settings.runtime_archive).is_file():
            raise FileNotFoundError(f"slurm.runtime_archive does not exist: {settings.runtime_archive}")
        for preflight in settings.preflight_files:
            if not Path(preflight).is_file():
                raise FileNotFoundError(f"slurm.preflight_files names a missing file: {preflight}")
        if not Path(str(config.ray.rendezvous_dir)).is_absolute():
            raise ValueError("ray.rendezvous_dir must be an absolute path on a filesystem shared by all nodes")
        require_environment(secret_names(config, settings, self._forward_env))
        return PreparedLaunch(config, settings, topology)

    def validate(self, config_path: Path) -> None:
        prepared = self._prepare(config_path)
        render_job(prepared, Path(prepared.settings.run_dir), self._forward_env)

    def launch(self, config_path: Path) -> LaunchOutcome:
        prepared = self._prepare(config_path)
        config, settings, topology = prepared.config, prepared.settings, prepared.topology
        run_dir = Path(settings.run_dir) / str(config.run.attempt_id)
        node_script, sbatch_script = render_job(prepared, run_dir, self._forward_env)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "launch.yaml").write_bytes(config_path.read_bytes())
        (run_dir / "node.sh").write_text(node_script)
        (run_dir / "job.sbatch").write_text(sbatch_script)
        job_id = parse_job_id(run_slurm_command(["sbatch", "--parsable", str(run_dir / "job.sbatch")]))
        logger.info(
            f"Submitted Slurm job {job_id}: {topology.num_nodes} node(s) x {topology.gpus_per_node} GPU, "
            f"logs {run_dir / 'logs'}"
        )
        if config.run.submission == SubmissionMode.DETACH:
            return LaunchOutcome(job_id=job_id, job_state=SUBMITTED_STATE, exit_code=0)
        with cancel_on_signal(job_id):
            outcome = wait_for_job(job_id, settings.poll_interval)
        logger.info(f"Slurm job {job_id} finished: {outcome.job_state} (exit {outcome.exit_code})")
        return outcome

    def export_terminal_policy(self, config_path: Path) -> None:
        raise NotImplementedError("Terminal policy export is not supported on Slurm yet; set run.export_hf=false")
