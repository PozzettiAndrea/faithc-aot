# faithc-aot

**FaithContour + Atom3d, prebuilt ahead-of-time (AOT) as a single CUDA wheel.**

This is a packaging fork that bundles two upstream projects so the whole FaithContour
mesh encode/decode (remesh) stack installs **prebuilt** — no runtime JIT compilation,
no `nvcc` required on the target machine:

| Vendored package | Upstream | License |
|---|---|---|
| `faithcontour` | [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC) (v1.5.0) | Apache-2.0 |
| `atom3d` | [Luo-Yihao/Atom3d](https://github.com/Luo-Yihao/Atom3d) (v0.1.0) | MIT |

## Why this fork exists

Upstream `atom3d` JIT-compiles its CUDA kernels on first import via
`torch.utils.cpp_extension.load(...)`, which needs the CUDA toolkit present at runtime and
pays a one-time compile. This fork compiles **every** CUDA module ahead of time via
`CUDAExtension` and ships the `.so`s in the wheel:

- `faithcontour._C` — `faithcontour/_csrc/{bindings.cpp,kernels.cu}`
- `atom3d.kernels.cumtv_cuda` — `atom3d/kernels/cumtv_kernels.cu`
- `atom3d.kernels.bvh_cuda` — `atom3d/kernels/bvh_kernels.cu`
- `atom3d.kernels.floodfill_cuda` — `atom3d/kernels/flood_fill_kernels.cu`

The only change to the vendored sources is in `atom3d/kernels/{__init__,bvh,flood_fill}.py`:
each kernel loader now imports its **prebuilt** submodule first and only falls back to JIT
for source installs.

## Install

Built and distributed via [cuda-wheels](https://github.com/PozzettiAndrea/cuda-wheels)
(package `faithc_aot`). Install the wheel for your torch/CUDA combo, plus `torch_scatter`
(not vendored — `faithcontour` has a pure-torch fallback, but the prebuilt `torch_scatter`
is faster and is its own cuda-wheels package):

```bash
pip install faithc-aot torch_scatter --index-url <cuda-wheels-index> --extra-index-url https://pypi.org/simple
```

`import faithcontour` and `import atom3d` then both work with zero runtime compilation.

> Do **not** also install the unrelated PyPI `atom3d` (drorlab protein/bio ML) into the same
> environment — it shares the `atom3d` import name and would shadow this one.
