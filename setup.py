"""
faithc-aot — FaithContour + Atom3d, prebuilt ahead-of-time (AOT).

Bundles two upstream projects into a single CUDA wheel so the whole FaithContour
remesh stack installs prebuilt, with no runtime JIT compilation and no nvcc on the
target machine:

  * faithcontour  (Luo-Yihao/FaithC)      — mesh encode/decode codec (Apache-2.0)
  * atom3d        (Luo-Yihao/Atom3d)       — CUDA mesh primitives (MIT)

Upstream `atom3d` JIT-compiles its kernels at first import; here every CUDA module
is compiled ahead of time via CUDAExtension and shipped in the wheel:

  * faithcontour._C                 (faithcontour/_csrc/{bindings.cpp,kernels.cu})
  * atom3d.kernels.cumtv_cuda       (atom3d/kernels/cumtv_kernels.cu)
  * atom3d.kernels.bvh_cuda         (atom3d/kernels/bvh_kernels.cu)
  * atom3d.kernels.floodfill_cuda   (atom3d/kernels/flood_fill_kernels.cu)

torch_scatter is NOT vendored — faithcontour falls back to a pure-torch shim if it
is absent, and it is available as its own prebuilt cuda-wheels package; install it
alongside for best performance.

Built by cuda-wheels (`pip wheel . --no-build-isolation --no-deps` with
TORCH_CUDA_ARCH_LIST set per build combo).
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

_nvcc = ["-O3", "--use_fast_math"]
_cxx = ["-O3"]


def _ext(name, sources):
    return CUDAExtension(
        name=name,
        sources=sources,
        extra_compile_args={"cxx": _cxx, "nvcc": _nvcc},
    )


ext_modules = [
    _ext("faithcontour._C", ["faithcontour/_csrc/bindings.cpp",
                             "faithcontour/_csrc/kernels.cu"]),
    _ext("atom3d.kernels.cumtv_cuda", ["atom3d/kernels/cumtv_kernels.cu"]),
    _ext("atom3d.kernels.bvh_cuda", ["atom3d/kernels/bvh_kernels.cu"]),
    _ext("atom3d.kernels.floodfill_cuda", ["atom3d/kernels/flood_fill_kernels.cu"]),
]

setup(
    name="faithc-aot",
    # Plain version (no local '+aot' segment): cuda-wheels appends its own
    # '+cuXXXtorchYY' local segment at rename time, and PEP 440 allows only one.
    version="1.5.0",
    description="FaithContour + Atom3d, prebuilt AOT CUDA wheel (mesh encode/decode for remeshing).",
    long_description=__doc__,
    long_description_content_type="text/plain",
    url="https://github.com/PozzettiAndrea/faithc-aot",
    license="Apache-2.0 AND MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "trimesh",
        "numpy",
        "scipy",
        "einops",
        "torch>=2.0.0",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # Ship atom3d's .cu so its JIT loaders still work on a source/torch mismatch.
    package_data={"atom3d": ["kernels/*.cu"]},
    include_package_data=True,
    zip_safe=False,
)
