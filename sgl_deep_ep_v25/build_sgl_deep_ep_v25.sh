#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_sgl_deep_ep_v25.sh <output-dir> <cuda-version> <architecture>

Build a CUDA 13 sgl-deep-ep-v25 wheel from the checkout containing this script.
The wheel installs DeepEP 2.5 as the `deep_ep_v25` Python package.
Build dependencies, including PyTorch and NCCL 2.32.3+ with headers, must already be installed.
EOF
}

if [[ $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

PACKAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPEP_SOURCE="$(cd "${PACKAGING_DIR}/.." && pwd)"
OUTPUT_DIR="$1"
CUDA_VERSION="$2"
ARCHITECTURE="$3"
PYTHON_BIN="${PYTHON_BIN:-python}"

# DeepEP 2.5 JIT-compiles its kernels with the host nvcc, which must be CUDA 13.1 or newer
case "${CUDA_VERSION}" in
    13.[1-9]|13.[1-9].*)
        CUDA_MINOR="${CUDA_VERSION#13.}"
        CUDA_TAG="cu13${CUDA_MINOR%%.*}"
        ;;
    *)
        echo "Unsupported CUDA version: ${CUDA_VERSION}; expected 13.1 or newer within CUDA 13" >&2
        exit 2
        ;;
esac

case "${ARCHITECTURE}" in
    x86_64|aarch64) ;;
    *)
        echo "Unsupported architecture: ${ARCHITECTURE}; expected x86_64 or aarch64" >&2
        exit 2
        ;;
esac

HOST_ARCH="$(uname -m)"
if [[ "${HOST_ARCH}" == arm64 ]]; then
    HOST_ARCH=aarch64
fi
if [[ "${HOST_ARCH}" != "${ARCHITECTURE}" ]]; then
    echo "Build architecture ${ARCHITECTURE} does not match container architecture ${HOST_ARCH}" >&2
    exit 1
fi

for required_file in setup.py deep_ep/__init__.py; do
    if [[ ! -f "${DEEPEP_SOURCE}/${required_file}" ]]; then
        echo "DeepEP source is missing ${required_file}: ${DEEPEP_SOURCE}" >&2
        exit 1
    fi
done

for required_file in __init__.py setup.py prerequisites.py rename_package.py README.md LICENSE VERSION pyproject.toml; do
    if [[ ! -f "${PACKAGING_DIR}/${required_file}" ]]; then
        echo "Packaging directory is missing ${required_file}: ${PACKAGING_DIR}" >&2
        exit 1
    fi
done

if ! command -v nvcc >/dev/null; then
    echo "nvcc is required to build sgl-deep-ep-v25" >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ ! -d "${CUDA_HOME}" ]]; then
    echo "CUDA_HOME does not exist: ${CUDA_HOME}" >&2
    exit 1
fi

CCCL_INCLUDE_DIR="${CUDA_HOME}/include/cccl"
if [[ ! -d "${CCCL_INCLUDE_DIR}" ]]; then
    echo "CUDA 13 CCCL headers were not found at ${CCCL_INCLUDE_DIR}" >&2
    exit 1
fi
export CPATH="${CCCL_INCLUDE_DIR}${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CCCL_INCLUDE_DIR}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export NVCC_PREPEND_FLAGS="-I${CCCL_INCLUDE_DIR}${NVCC_PREPEND_FLAGS:+ ${NVCC_PREPEND_FLAGS}}"

if [[ -n "${SGL_DEEP_EP_V25_VERSION:-}" ]]; then
    PUBLIC_VERSION="${SGL_DEEP_EP_V25_VERSION}"
else
    PUBLIC_VERSION="$(<"${PACKAGING_DIR}/VERSION")"
fi
if [[ -z "${PUBLIC_VERSION}" ]]; then
    echo "sgl-deep-ep-v25 version is empty" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
STAGING_DIR="$(mktemp -d -t sgl-deep-ep-v25-build.XXXXXX)"
cleanup() {
    rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

# Exclude artifacts from earlier builds (including the usual output paths).
# The source checkout may be read-only, as it is in the release container.
tar -C "${DEEPEP_SOURCE}" --exclude=.git --exclude=build --exclude=dist \
    --exclude=dist-pypi --exclude='*.egg-info' --exclude=__pycache__ \
    -cf - . | tar -C "${STAGING_DIR}" -xf -
mv "${STAGING_DIR}/setup.py" "${STAGING_DIR}/_deepep_setup.py"
cp "${PACKAGING_DIR}/setup.py" "${STAGING_DIR}/setup.py"
cp "${PACKAGING_DIR}/pyproject.toml" "${STAGING_DIR}/pyproject.toml"
cp "${PACKAGING_DIR}/README.md" "${STAGING_DIR}/README.md"
cp "${PACKAGING_DIR}/LICENSE" "${STAGING_DIR}/LICENSE"

# Install as `deep_ep_v25` so this wheel can coexist with sgl-deep-ep (`deep_ep`)
"${PYTHON_BIN}" "${PACKAGING_DIR}/rename_package.py" "${STAGING_DIR}"
PACKAGE_DIR="${STAGING_DIR}/deep_ep_v25"

cp "${PACKAGING_DIR}/prerequisites.py" "${PACKAGE_DIR}/prerequisites.py"

cat >"${PACKAGE_DIR}/_build_info.py" <<EOF
EXPECTED_CUDA_MAJOR = 13
MIN_CUDA_MINOR = 1
CUDA_TAG = "${CUDA_TAG}"
EOF

ORIGINAL_INIT="${PACKAGE_DIR}/__init__.py"
GUARDED_INIT="${PACKAGE_DIR}/__init__.py.guarded"
cp "${PACKAGING_DIR}/__init__.py" "${GUARDED_INIT}"
printf '\n\n' >>"${GUARDED_INIT}"
cat "${ORIGINAL_INIT}" >>"${GUARDED_INIT}"
mv "${GUARDED_INIT}" "${ORIGINAL_INIT}"

# DeepEP bakes these variables into the wheel as import-time defaults, so keep build-host
# values out of it. A build-time NCCL root is still honored through the non-persistent NCCL_DIR.
if [[ -n "${EP_NCCL_ROOT_DIR:-}" ]]; then
    export NCCL_DIR="${EP_NCCL_ROOT_DIR}"
fi
unset EP_JIT_CACHE_DIR EP_JIT_PRINT_COMPILER_COMMAND EP_JIT_CPP_STANDARD EP_NUM_TOPK_IDX_BITS \
    EP_NCCL_ROOT_DIR EP_DEFAULT_RDMA_SL EP_OVERRIDE_RDMA_SL

export CUDA_HOME
export MAX_JOBS="${MAX_JOBS:-8}"
export SGL_DEEP_EP_V25_VERSION="${PUBLIC_VERSION}+${CUDA_TAG}"

rm -rf "${STAGING_DIR}/build" "${STAGING_DIR}/dist"
(
    cd "${STAGING_DIR}"
    "${PYTHON_BIN}" setup.py bdist_wheel --dist-dir dist
)

shopt -s nullglob
WHEELS=("${STAGING_DIR}"/dist/*.whl)
if [[ ${#WHEELS[@]} -ne 1 ]]; then
    echo "Expected exactly one wheel, found ${#WHEELS[@]}" >&2
    exit 1
fi

cp "${WHEELS[0]}" "${OUTPUT_DIR}/"
echo "Built ${OUTPUT_DIR}/$(basename "${WHEELS[0]}")"
