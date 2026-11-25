# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the sbatch script and the per-node script for one MarinSkyRL launch.

Option values come from launch configs, so each one is checked against a conservative
character set before it reaches a script. Secret values are never rendered.
"""

import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SAFE_VALUE = re.compile(r"[A-Za-z0-9_.:@/+,%=-]+")
SBATCH_OPTION = re.compile(r"--[a-z][a-z-]*(=[A-Za-z0-9_.:@/+,%=-]+)?")
TIME_LIMIT = re.compile(r"(\d+-)?\d{1,2}(:\d{2}){0,2}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NETWORK_INTERFACE = re.compile(r"[A-Za-z0-9_.:-]+")


@dataclass(frozen=True)
class SbatchSpec:
    job_name: str
    nodes: int
    gpus_per_node: int
    time_limit: str
    log_dir: str
    node_script: str
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    reservation: str | None = None
    constraint: str | None = None
    cpus_per_task: int | None = None
    extra_options: tuple[str, ...] = ()
    broadcast_archive: str | None = None
    node_local_dir: str = "/tmp"


@dataclass(frozen=True)
class NodeScriptSpec:
    forwarded_env: tuple[str, ...]
    env: Mapping[str, str]
    ray_port_vars: tuple[str, ...]
    advertise_interface: str | None
    body: str
    preflight_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimePaths:
    runtime_env: str
    activation_file: str
    repo_root: str
    archive: str | None = None
    local_dir: str = "/tmp"
    broadcast: bool = False
    """The archive was already sent to ``archive_copy(local_dir)`` on every node by the batch script."""


PREFLIGHT_BYTES = 32 * 1024 * 1024
PREFLIGHT_SECONDS = 180


def archive_copy(local_dir: str) -> str:
    """Node-local path the batch script broadcasts the runtime archive to (``${USER}`` is expanded by the shell)."""
    return f"{_checked('node-local directory', local_dir, SAFE_VALUE)}/skyrl-${{USER}}-archive.tar"


def _checked(name: str, value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Unsafe or malformed {name}: {value!r}")
    return value


def render_sbatch(spec: SbatchSpec) -> str:
    """Return a batch script that starts one node script per allocated node."""
    options = {
        "job-name": _checked("job name", spec.job_name, SAFE_VALUE),
        "nodes": str(spec.nodes),
        "ntasks-per-node": "1",
        "gpus-per-node": str(spec.gpus_per_node),
        "time": _checked("time limit", spec.time_limit, TIME_LIMIT),
        "output": f"{_checked('log directory', spec.log_dir, SAFE_VALUE)}/%x-%j.out",
        "open-mode": "append",
        "export": "ALL",
    }
    optional = {
        "partition": spec.partition,
        "account": spec.account,
        "qos": spec.qos,
        "reservation": spec.reservation,
        "constraint": spec.constraint,
        "cpus-per-task": None if spec.cpus_per_task is None else str(spec.cpus_per_task),
    }
    for name, value in optional.items():
        if value is not None:
            options[name] = _checked(name, value, SAFE_VALUE)
    lines = ["#!/bin/bash", *(f"#SBATCH --{name}={value}" for name, value in options.items()), "#SBATCH --exclusive"]
    lines += [f"#SBATCH {_checked('sbatch option', option, SBATCH_OPTION)}" for option in spec.extra_options]
    node_script = _checked("node script path", spec.node_script, SAFE_VALUE)
    lines += ["set -euo pipefail", 'echo "slurm job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"']
    if spec.broadcast_archive is not None:
        archive = _checked("runtime archive", spec.broadcast_archive, SAFE_VALUE)
        # One read of the archive on the batch host; slurmd fans it out to every node's local disk.
        lines.append(f'sbcast --force {archive} "{archive_copy(spec.node_local_dir)}"')
    lines += [
        # exec lets a cancellation signal reach srun directly. Without --cpu-bind=none a step of one task per node is
        # bound to a single core under TaskPluginParam=cores, and every process of the job shares that core.
        f'exec srun --kill-on-bad-exit=1 --ntasks="${{SLURM_JOB_NUM_NODES}}" --ntasks-per-node=1 --cpu-bind=none --label bash {node_script}',
    ]
    return "\n".join(lines) + "\n"


def render_node_script(spec: NodeScriptSpec) -> str:
    """Return the script every node runs: map Slurm ranks onto the launcher's task variables, then run ``body``.

    The runtime reads its rank, task count and Ray endpoints from the same variables Iris sets. A zero
    port placeholder makes it pick a free port.
    """
    # No -u: the sourced runtime file may reference unset variables.
    lines = ["#!/usr/bin/env bash", "set -eo pipefail", render_clean_environment(spec.forwarded_env)]
    lines += ['export IRIS_TASK_ID="${SLURM_PROCID}"', 'export IRIS_NUM_TASKS="${SLURM_NTASKS}"']
    # A step bound to one core makes everything slow without an error; this line shows it in the job log.
    lines.append('echo "[slurm] rank ${SLURM_PROCID} may use CPUs $(grep Cpus_allowed_list /proc/self/status | cut -f2)"')
    for file in spec.preflight_files:
        path = _checked("preflight file", file, SAFE_VALUE)
        lines.append(
            f'timeout {PREFLIGHT_SECONDS} head -c {PREFLIGHT_BYTES} -- {path} > /dev/null || {{ echo "[slurm] rank '
            f'${{SLURM_PROCID}} on $(hostname) could not read {PREFLIGHT_BYTES} bytes of {path} in {PREFLIGHT_SECONDS} s" >&2; exit 1; }}'
        )
    lines += [f"export {_checked('port variable', name, ENV_NAME)}=0" for name in spec.ray_port_vars]
    if spec.advertise_interface is not None:
        interface = _checked("network interface", spec.advertise_interface, NETWORK_INTERFACE)
        lines.append(
            f"export IRIS_ADVERTISE_HOST=\"$(ip -4 -o addr show dev {interface} | awk '{{print $4}}' | cut -d/ -f1 | head -n1)\""
        )
    lines += [f"export {_checked('environment name', name, ENV_NAME)}={shlex.quote(value)}" for name, value in spec.env.items()]
    lines.append(spec.body)
    return "\n".join(lines) + "\n"


HOST_ENVIRONMENT = ("HOME", "USER", "LOGNAME", "LANG", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
CLEAN_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def render_clean_environment(forwarded_env: Sequence[str]) -> str:
    """Shell lines that re-run the node script in a clean environment.

    The job inherits the submitting shell, and site modules set variables such as CC that break tools like Triton.
    The clean environment keeps the host basics, every SLURM_* variable and the forwarded secrets by name.
    """
    names = [_checked("environment name", name, ENV_NAME) for name in (*HOST_ENVIRONMENT, *forwarded_env)]
    return "\n".join(
        [
            'if [ -z "${SKYRL_CLEAN_ENV:-}" ]; then',
            f"  keep=({' '.join(names)})",
            "  args=()",
            '  for name in "${keep[@]}" $(compgen -v SLURM_); do',
            '    if [ -n "${!name+x}" ]; then args+=("$name=${!name}"); fi',
            "  done",
            f'  exec env -i "${{args[@]}}" PATH={CLEAN_PATH} TMPDIR=/tmp SKYRL_CLEAN_ENV=1 bash "$0"',
            "fi",
        ]
    )


def parse_job_id(sbatch_output: str) -> str:
    """Return the job id from ``sbatch --parsable`` output (``ID`` or ``ID;cluster``).

    Site submit hooks can print banners before the id, so only the last line counts.
    """
    lines = [line for line in sbatch_output.splitlines() if line.strip()]
    job_id = lines[-1].strip().split(";", 1)[0] if lines else ""
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch output: {sbatch_output!r}")
    return job_id


def render_runtime_paths(paths: RuntimePaths) -> str:
    """Shell lines that set SKYRL_ENV, SKYRL_ACTIVATION and SKYRL_REPO for the node program.

    Without an archive they name the shared copies. With one, the lines first extract the environment and the
    repository to node-local disk and point at those copies, which are removed when the shell exits.

    The staged path is fixed per user, not per job: FlashInfer's JIT cache records absolute paths, so a path that
    changed with every job made each job recompile its kernels. Jobs get whole nodes, so only one uses the path at a time.
    """
    env = _checked("runtime environment", paths.runtime_env, SAFE_VALUE)
    activation = _checked("activation file", paths.activation_file, SAFE_VALUE)
    repo = _checked("repository root", paths.repo_root, SAFE_VALUE).rstrip("/")
    if paths.archive is None:
        return "\n".join([f"SKYRL_ENV={env}", f"SKYRL_ACTIVATION={activation}", f"SKYRL_REPO={repo}"])
    archive = _checked("runtime archive", paths.archive, SAFE_VALUE)
    local_dir = _checked("node-local directory", paths.local_dir, SAFE_VALUE)
    repo_parent, repo_name = posixpath.split(repo)
    source = f'"{archive_copy(local_dir)}"' if paths.broadcast else archive
    cleanup = f'"$SKYRL_STAGE" "{archive_copy(local_dir)}"' if paths.broadcast else '"$SKYRL_STAGE"'
    return "\n".join(
        [
            f'SKYRL_STAGE="{local_dir}/skyrl-${{USER}}"',
            # A killed earlier job may have left a copy behind.
            'rm -rf "$SKYRL_STAGE"',
            f"trap 'rm -rf {cleanup}' EXIT",
            'mkdir -p "$SKYRL_STAGE/env/lib"',
            'ln -sfn lib "$SKYRL_STAGE/env/lib64"',
            f'tar -xf {source} -C "$SKYRL_STAGE/env"',
            f'cp -a {env}/bin {env}/pyvenv.cfg "$SKYRL_STAGE/env/"',
            # Console scripts name the original environment in their shebang.
            f'for script in "$SKYRL_STAGE"/env/bin/*; do [ -f "$script" ] && [ ! -L "$script" ]'
            f' && sed -i "1s|^#!{env}/|#!$SKYRL_STAGE/env/|" "$script"; done',
            f'sed "s|{env}|$SKYRL_STAGE/env|g" {activation} > "$SKYRL_STAGE/runtime.env"',
            f'tar --exclude=.git -C {repo_parent} -cf - {repo_name} | tar -C "$SKYRL_STAGE" -xf -',
            'SKYRL_ENV="$SKYRL_STAGE/env"',
            'SKYRL_ACTIVATION="$SKYRL_STAGE/runtime.env"',
            f'SKYRL_REPO="$SKYRL_STAGE/{repo_name}"',
        ]
    )
