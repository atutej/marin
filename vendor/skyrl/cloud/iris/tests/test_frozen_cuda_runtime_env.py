"""The frozen GPU runtime's CUDA paths must resolve on hosts whose Python uses lib64 site-packages."""

from pathlib import Path

from marinskyrl.environment_contract import (
    CUDA_WHEEL_NAMESPACE,
    LD_LIBRARY_PATH_ENV,
    NVRTC_HOME_ENV,
    EnvVarManager,
    EnvVarScope,
)


def test_cuda_wheels_reached_through_lib_and_lib64_resolve_to_one_root(tmp_path: Path) -> None:
    # Red Hat family Pythons report both lib64 and lib site-packages; in a venv lib64 is a symlink to lib.
    venv = tmp_path / "venv"
    wheels = venv / "lib" / "python3.12" / "site-packages" / "nvidia"
    (wheels / CUDA_WHEEL_NAMESPACE / "lib").mkdir(parents=True)
    (wheels / "cudnn" / "lib").mkdir(parents=True)
    (venv / "lib64").symlink_to("lib")
    site_packages = [str(venv / "lib64" / "python3.12" / "site-packages"), str(venv / "lib" / "python3.12" / "site-packages")]

    environment = EnvVarManager.for_frozen_cuda_runtime(site_packages).environment_for(EnvVarScope.TASK_RUNTIME)

    assert Path(environment[NVRTC_HOME_ENV]).resolve() == (wheels / CUDA_WHEEL_NAMESPACE).resolve()
    library_dirs = [Path(path).resolve() for path in environment[LD_LIBRARY_PATH_ENV].split(":")]
    assert sorted(library_dirs) == sorted([(wheels / CUDA_WHEEL_NAMESPACE / "lib").resolve(), (wheels / "cudnn" / "lib").resolve()])
