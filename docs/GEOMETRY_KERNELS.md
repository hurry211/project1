# Element stiffness on regular and distorted meshes

The verifier integrates local stiffness matrices independently of MFEM. Its streaming path can contract gradients without assuming a Cartesian mesh. On a linear triangle the physical basis gradients are constant across the cell, so the implementation sums the quadrature weights times the diffusivity before forming the gradient product. On a bilinear quadrilateral it retains the gradient at each quadrature point. Both paths evaluate variable diffusivity at the mapped quadrature points.

The 2026-11-25 WSL2 experiment generated five native MFEM fields and compared each cell matrix from full quadrature with the corresponding streaming result. Each arm processed the same batches and the same 36 quadrature points per cell. One warm-up pair preceded eight scored pairs per case, with each arm running first four times. NumPy 2.5.5 used one CPU and one BLAS thread. Timings cover local stiffness only. Basis and mapped quadrature construction, sparse assembly, period-based reuse and the native solve are outside the timed intervals. [Benchmark code](../benchmarks/geometry_kernels.py) · [All timings and source hashes](../evidence/e4/geometry.json)

| Mesh and diffusivity | Cells | Detected period | Full median (ms) | Streaming median (ms) | Full / streaming | Largest cell difference* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cartesian P1 triangle, 1 | 32,768 | 2 | 37.03 | 10.83 | 3.42 | 2.23e−16 |
| Cartesian Q1 quad, 1 | 16,384 | 1 | 226.53 | 54.63 | 4.15 | 1.34e−15 |
| Cartesian P1 triangle, 1+x | 32,768 | 2 | 62.50 | 25.63 | 2.44 | 9.46e−16 |
| Unstructured P1 triangle, 1 | 8,192 | 8,192 | 9.81 | 3.35 | 2.93 | 1.90e−15 |
| Distorted Q1 quad, 1 | 4,096 | 4,096 | 51.90 | 12.45 | 4.17 | 1.27e−15 |

\* For each cell, the largest absolute entry difference is divided by the largest absolute reference stiffness entry. Every cell must meet the limit of 64 float64 machine epsilons. Both implementations also produced identical bytes between repeated executions of their own arm. The differing accumulation orders permit roundoff between arms. Separate regressions cover variable diffusivity on distorted triangles and quadrilaterals. [Regression tests](../tests/test_geometry_kernels.py)

## Reproduce

Use a new output directory for the native fields and a new report path. Set `CPU` to one available Linux CPU and `POISSON_BIN` to the native serial driver.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c "$CPU" python -m benchmarks.geometry_kernels \
  --directory "$RUN/geometry" --binary "$POISSON_BIN" --blocks 8 --report "$RUN/geometry.json"
```
