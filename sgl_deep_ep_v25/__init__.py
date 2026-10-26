from ._build_info import EXPECTED_CUDA_MAJOR as _EXPECTED_CUDA_MAJOR, MIN_CUDA_MINOR as _MIN_CUDA_MINOR
from .prerequisites import check_prerequisites as _check_prerequisites

if _EXPECTED_CUDA_MAJOR is not None:
    _check_prerequisites(expected_cuda_major=_EXPECTED_CUDA_MAJOR, min_cuda_minor=_MIN_CUDA_MINOR)
del _EXPECTED_CUDA_MAJOR, _MIN_CUDA_MINOR, _check_prerequisites
