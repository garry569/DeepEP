# sgl-deep-ep-v25

`sgl-deep-ep-v25` is SGLang's binary distribution of DeepEP 2.5
([deepseek-ai/DeepEP#763](https://github.com/deepseek-ai/DeepEP/pull/763)), built
from the `sgl-deepep-v25` branch of [sgl-project/DeepEP](https://github.com/sgl-project/DeepEP).
It installs the Python package `deep_ep_v25`:

```python
import deep_ep_v25

buffer = deep_ep_v25.EPBuffer(...)
```

DeepEP 2.5 removes the V1 `Buffer` API and NVSHMEM, and replaces `ElasticBuffer`
with `EPBuffer`, `EngramBuffer`, `PPBuffer`, and `BucketBuffer`. The V1 and
DeepEP 2.1 APIs remain available in the separate
[`sgl-deep-ep`](https://github.com/sgl-project/DeepEP/tree/sgl-deepep/sgl_deep_ep)
package, which installs `deep_ep`. The two packages have different import
names and can be installed in the same environment.

## Installation

```bash
pip install sgl-deep-ep-v25
```

Each wheel is specific to a CPython version, CPU architecture, and CUDA major.
The package checks the host prerequisites below when `deep_ep_v25` is imported.

## Host prerequisites

The wheel contains DeepEP's Python code and a small host extension. DeepEP 2.5
compiles its CUDA kernels at runtime with DeepJIT, so every host must provide:

- Linux on x86_64 or aarch64, a compatible NVIDIA driver, and a CUDA 13 PyTorch build;
- a CUDA toolkit with `nvcc` 13.1 or newer, found through `CUDA_HOME`,
  `CUDA_PATH`, `nvcc` on `PATH`, or `/usr/local/cuda` (or `EP_JIT_NVCC_COMPILER`),
  and a host compiler with C++20 support;
- NCCL 2.32.3 or newer, including its headers. DeepEP finds it through
  `EP_NCCL_ROOT_DIR`, `NCCL_DIR`, or the `nvidia-nccl-cu13` Python package;
- the RDMA/NVLink environment required by the selected transport.

PyTorch may pin an older `nvidia-nccl-cu13`. In that case, install the newer NCCL
over it, for example `pip install --no-deps "nvidia-nccl-cu13>=2.32.3"`.
At import time, DeepEP also checks that the process loads exactly one NCCL
library and that it matches the one found above; `EP_SUPPRESS_NCCL_CHECK=1`
skips that check.

Compiled kernels are cached under `$HOME/.dj` by default (`EP_JIT_CACHE_DIR`).
See the [DeepEP README](../README.md#environment-variables) for the other
runtime variables.

## Wheel release workflow

Wheels are built and published by SGLang's `release-whl-deepep-v25.yml`
workflow. Implementation and packaging are maintained together on
`sgl-deepep-v25`, which tracks upstream DeepEP `main` with these additions:

- `sgl_deep_ep_v25/`: this packaging directory;
- the manylinux `pidfd` syscall fix in `csrc/kernels/comm/symmetric.hpp`.

Run the workflow with the public package version (without a leading `v`) and
the `deepep-ref` to build (normally `sgl-deepep-v25`; use a commit for a
reproducible release). Then install the wheel in a clean environment and verify
both `import deep_ep_v25` and a multi-GPU DeepEP or SGLang workload.

## Building from a checkout

Initialize the DeepJIT submodule and install the build dependencies: the target
CUDA 13 PyTorch build, a CUDA 13.1+ toolkit, NCCL 2.32.3+ with headers, ninja,
setuptools, and wheel. NVSHMEM is not needed. Then run:

```bash
git clone --recursive --branch sgl-deepep-v25 https://github.com/sgl-project/DeepEP.git
cd DeepEP
SGL_DEEP_EP_V25_VERSION=0.1.0 bash sgl_deep_ep_v25/build_sgl_deep_ep_v25.sh dist 13.1 x86_64
```

The arguments are the output directory, CUDA version (13.1 or newer), and CPU
architecture (`x86_64` or `aarch64`). The script stages a temporary copy of the
checkout, renames the `deep_ep` Python package to `deep_ep_v25` there
(`rename_package.py`), and builds the wheel; the source checkout is unchanged.
The C++ sources and the `include/deep_ep/` header tree keep their names.

`SGL_DEEP_EP_V25_VERSION` overrides `sgl_deep_ep_v25/VERSION`; the wheel
version gets a `+cu13X` suffix for the build's CUDA minor version.
`PYTHON_BIN` selects the build interpreter and `MAX_JOBS` controls parallelism.
DeepEP stores some `EP_*` variables in the wheel as import-time defaults, so the
script unsets them for the build; `EP_NCCL_ROOT_DIR` is passed on as `NCCL_DIR`.
