import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

_README_URL = ("https://github.com/sgl-project/DeepEP/tree/sgl-deepep-v25/sgl_deep_ep_v25")
_MIN_NCCL_VERSION = (2, 32, 3)


def _failure(message: str) -> ImportError:
    return ImportError(f"{message}. See {_README_URL} for host prerequisites")


def _find_nvcc() -> str:
    # Follow DeepJIT's discovery order: explicit compiler, then CUDA_HOME, CUDA_PATH, PATH and /usr/local/cuda
    for name in ("EP_JIT_NVCC_COMPILER", "DJ_JIT_NVCC_COMPILER"):
        if os.environ.get(name):
            return os.environ[name]
    for name in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(name):
            return str(Path(os.environ[name]) / "bin" / "nvcc")
    return shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def _check_nvcc(expected_cuda_major: int, min_cuda_minor: int) -> None:
    nvcc = _find_nvcc()
    try:
        output = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise _failure(f"DeepEP compiles kernels at runtime and needs nvcc, but {nvcc!r} is not usable") from error

    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        raise _failure(f"Cannot parse the CUDA version reported by {nvcc!r}")
    version = (int(match.group(1)), int(match.group(2)))
    if version[0] != expected_cuda_major or version[1] < min_cuda_minor:
        raise _failure(f"DeepEP 2.5 needs nvcc {expected_cuda_major}.{min_cuda_minor} or newer within CUDA {expected_cuda_major}, "
                       f"but {nvcc!r} is {version[0]}.{version[1]}")


def _check_nccl() -> None:
    from .utils.find_pkgs import find_nccl_root

    nccl_root = find_nccl_root(optional=True)
    if nccl_root is None:
        raise _failure("Cannot find NCCL; install nvidia-nccl-cu13 or set EP_NCCL_ROOT_DIR")

    # DeepEP JIT-compiles against the NCCL headers, so they are needed at runtime as well
    header = Path(nccl_root) / "include" / "nccl.h"
    if not header.is_file():
        raise _failure(f"NCCL headers are missing: {header}")

    defines = dict(re.findall(r"^#define NCCL_(MAJOR|MINOR|PATCH)\s+(\d+)", header.read_text(), re.MULTILINE))
    try:
        version = tuple(int(defines[key]) for key in ("MAJOR", "MINOR", "PATCH"))
    except KeyError as error:
        raise _failure(f"Cannot parse the NCCL version from {header}") from error
    if version < _MIN_NCCL_VERSION:
        raise _failure(f"DeepEP 2.5 needs NCCL {'.'.join(map(str, _MIN_NCCL_VERSION))} or newer, "
                       f"but {nccl_root} has {'.'.join(map(str, version))}")


def check_prerequisites(*, expected_cuda_major: int, min_cuda_minor: int) -> None:
    if platform.system() != "Linux":
        raise _failure("sgl-deep-ep-v25 supports Linux only")

    machine = platform.machine()
    if machine not in {"x86_64", "aarch64"}:
        raise _failure(f"sgl-deep-ep-v25 does not support architecture {machine}")

    try:
        import torch
    except ImportError as error:
        raise _failure("PyTorch is not installed") from error

    torch_cuda = torch.version.cuda
    if not torch_cuda:
        raise _failure("The installed PyTorch build does not provide CUDA")

    try:
        torch_cuda_major = int(torch_cuda.split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise _failure(f"Cannot parse PyTorch CUDA version {torch_cuda!r}") from error

    if torch_cuda_major != expected_cuda_major:
        raise _failure(f"This wheel requires CUDA {expected_cuda_major}, but PyTorch uses CUDA {torch_cuda_major}")

    if not torch.cuda.is_available():
        raise _failure("The NVIDIA driver does not expose a usable CUDA device")

    _check_nvcc(expected_cuda_major, min_cuda_minor)
    _check_nccl()
