# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``slurm`` section of a SkyRL launch config."""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class SlurmConfig:
    """Slurm placement and runtime-environment settings for one launch.

    Node and GPU counts come from ``iris.allocation``, which the role plan already validates.
    ``runtime_env`` and ``activation_file`` are produced by ``cloud/iris/bootstrap_runtime.sh``;
    the launcher activates them and never builds them.
    """

    run_dir: str = MISSING
    """Shared-filesystem directory for rendered scripts, the launch config copy and job logs."""
    time_limit: str = MISSING
    """Value for ``sbatch --time``, for example ``12:00:00`` or ``1-00:00:00``."""
    runtime_env: str = MISSING
    """Environment directory created by ``bootstrap_runtime.sh`` (contains ``bin/python``)."""
    activation_file: str = MISSING
    """Runtime file written by ``bootstrap_runtime.sh``; sourced on every node."""
    runtime_archive: str | None = None
    """Tarball of the runtime environment's ``lib`` directory, made with ``tar -chf ARCHIVE -C RUNTIME_ENV lib``.
    It must include the linker names that ``bootstrap_runtime.sh`` adds beside site-packages, for example
    ``libcudart.so``. When set, every node extracts it and the repository checkout into ``node_local_dir`` before
    starting, because imports from a shared filesystem are slow."""
    broadcast_runtime_archive: bool = True
    """Send ``runtime_archive`` to every node with ``sbcast`` from the batch script, so the shared filesystem serves it
    once per job and not once per node. Jobs that start together otherwise all pull it through the same file server."""
    preflight_files: list[str] = field(default_factory=list)
    """Files each node reads the first 32 MiB of, with a 180 s limit, before staging anything. A node that cannot read
    them (for example model weights on a stalled mount) fails the job within minutes and names itself."""
    node_local_dir: str = "/tmp"
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    reservation: str | None = None
    constraint: str | None = None
    cpus_per_task: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    """Non-secret environment variables exported on every node, for example NCCL settings. The node program starts
    from a clean environment, so anything it needs beyond Slurm's own variables must be listed here."""
    forward_env: list[str] = field(default_factory=list)
    """Names of secrets taken from the submitting environment. Values are never written to disk."""
    extra_sbatch: list[str] = field(default_factory=list)
    """Additional ``sbatch`` options such as ``--exclude=c101-001``."""
    advertise_interface: str | None = None
    """Interface whose IPv4 address Ray advertises. Unset uses the default-route address."""
    poll_interval: int = 30
