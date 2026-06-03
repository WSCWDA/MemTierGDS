from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except ImportError as exc:
    raise RuntimeError(
        "MemTier PyTorch requires PyTorch to build. Install a CUDA-enabled torch "
        "wheel before running `pip install -e ./memtier_pytorch`."
    ) from exc

if CUDA_HOME is None:
    raise RuntimeError(
        "MemTier PyTorch requires a CUDA toolkit to build. "
        "Install CUDA and ensure nvcc/CUDA_HOME are available."
    )

ROOT = Path(__file__).parent

setup(
    name="memtier-pytorch",
    version="0.1.0",
    description="Minimal MemTier CUDA UVM file/data prototype for PyTorch",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="memtier.memtier_ext",
            sources=[
                str(ROOT / "csrc" / "memtier_ext.cpp"),
                str(ROOT / "csrc" / "memtier_uvm.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "numpy"],
)
