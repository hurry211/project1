# MFEM 科学计算智能体阶段汇报

李老师您好，

这次我围绕两个问题做了复现：在解达到同一精度且通过独立验证的前提下，哪一部分计算值得优化？对常规 Poisson 以外的问题，怎样证明 MFEM 算出了正确的场？复现对象是性能验证分支的 `903bb0b`。它已包含上游 `ceb485c`。这份汇报只整理该版本和它的历史实验。[项目初步规划](ORIGINAL_PROJECT_PROPOSAL_CN.md) · [本次证据索引](../evidence/stage-review/index.json)

## 1. 重试时可以省掉十二次重复求解

在同一正弦 Poisson 精度任务上，逐次加密并重算全部层级需要调用 MFEM 18 次。复用通过独立验证的已有层级后，只需求解 6 次。新尝试保存完整副本，逐文件核对摘要，并对全部层级重新判定收敛阶和精度。单独重放时，验证器从原始网格和场值重新计算。两种策略的最细网格 L² 误差均为 8.4522×10⁻⁵，迭代次数均为 138。各预热一次、交替顺序正式测量三对，端到端耗时中位数从 2.781 秒降到 2.543 秒。逐对计算的耗时比为 1.083–1.149，中位数 1.139。[重试逐次测量](../evidence/stage-review/retry.json.gz) · [重试实现](../sciagent/controller.py)

## 2. 部分装配的收益取决于阶次和所计入的环节

我在混合边界制造解上固定了 CG、Jacobi、收敛容差、网格序列（16、32、64）和独立验证条件，只改变装配方式。每组舍弃一次预热，再做三对交替顺序的测量。下表为各列中位数。装配、预条件器设置及求解时间累加了三级网格，迭代次数和 L² 误差取最细级。两列内存分别为求解、验证进程峰值，单位 MiB。[逐次计时、误差、内存及离散程度](../evidence/stage-review/performance.json.gz)

| 阶次与装配 | 装配 ms | 预条件器 ms | CG ms | 验证 s | 端到端 s | 求解 / 验证 MiB | 迭代 | L² 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 全装配 | 3.67 | 0.057 | 4.36 | 0.521 | 0.781 | 10.7 / 89.5 | 152 | 2.545×10⁻⁴ |
| Q1 部分装配 | 4.86 | 0.183 | 28.48 | 0.527 | 0.812 | 12.2 / 89.6 | 152 | 2.545×10⁻⁴ |
| Q2 全装配 | 20.65 | 0.238 | 70.41 | 1.017 | 1.459 | 18.9 / 139.1 | 358 | 8.600×10⁻⁷ |
| Q2 部分装配 | 6.50 | 0.551 | 150.07 | 0.953 | 1.427 | 14.9 / 139.3 | 358 | 8.600×10⁻⁷ |

Q2 部分装配节省了矩阵装配时间和求解进程内存，但 CG 用时超过全装配的两倍。三对端到端耗时比（全装配除以部分装配）从 0.948 到 1.093，快慢次序发生变化。Q1 的三对测量均是全装配较快。两个阶次下，独立验证都占端到端耗时的大半。对当前串行硬件，继续降低验证成本，比一律切换装配方式更值得检验。

## 3. 特殊算例已有可判错的数学基准

| 算例 | 当前可验证的范围 | 判据 |
|---|---|---|
| 各向异性扩散 | 单位方形、常张量 diag(1000,1)、全 Dirichlet、Q1 全装配 CG+Jacobi | 将正弦解析解代入张量扩散算子，核对源项、误差和独立离散解 |
| 双层材料扩散 | 界面对齐、材料系数 1 与 10,000、左右单位势差、上下绝热、Q1 | 逐层核对分段线性解析场、界面两侧通量和总能量。把材料改均匀、把零源项改成单位载荷，两组都应被拒绝 |
| 空间变系数扩散 | κ=1+x，生成网格或受审查的 Gmsh 2.2 单位方形网格 | 检查源项中的系数导数、边界通量和独立参考解，并校验输入网格摘要 |

本次从原始场重放五份历史归档，得到三个预期接受和两个预期拒绝，失败码与归档一致。另以 scikit-fem 作对照，三种标准制造解、两种网格来源、七种离散与求解配置的 42 项组合全部通过。这 42 项的覆盖范围与表中前两种特殊算例各有对应的测试条件。[五份旧归档重放](../evidence/stage-review/portable.json) · [42 项独立交叉验证](../evidence/stage-review/stage2.json) · [高对比度算例的原始测量](../evidence/scaling-stage/layered-evidence.json)

## 4. 验收、策略与证据范围

- **回归与判错。** 锁定 MFEM 4.10 源码提交，在 Ubuntu 24.04 WSL2 下分别通过 433 项串行环境测试和 9 项 MPI 环境测试。31 组历史任务的验收状态、冻结输入和重试决策全部匹配，已接受任务的误差范数最大相对差为零。验证器挑战的 284 项试验和 14 项参考求解器故障对照均得到预期结论。[串行测试](../evidence/stage-review/tests.log) · [MPI 测试](../evidence/stage-review/mpi-tests.log) · [历史任务](../evidence/stage-review/legacy.json) · [故障对照](../evidence/stage-review/challenge.log)
- **确定性选型。** 先按能力矩阵和自由度预算筛选，再按冻结精度与独立验收筛选，最后比较可行候选的重复测量成本。本次规则规划器生成了变系数 Gmsh 任务，完成 MFEM 求解并通过摘要重放，模型调用数为零。上游另有本地模型实测，本次未重跑模型推理。[规则规划重放](../evidence/stage-review/planner-replay.json) · [本地模型历史记录](../evidence/stage4/replication-results.json)
- **并行范围。** 对正弦解 P1 三角形，实际启动 1、2、4 个 MPI 进程，完成 12 组进程数与网格组合、12 个计时样本，并通过归档核验。这个实验给出了该问题的并行正确性证据。GPU 性能以及特殊算例的并行表现仍待测量。[MPI 重放回执](../evidence/stage-review/mpi-replay.json)
- **上游历史性能。** E1 有 396 次合格试验、396 份清单核查和 12 次代表性数值重放。E2 在相同归档上分别做 30 对测量，大规模正弦场的验证时间中位数从 18.50 秒降至 11.70 秒，混合边界场从 0.775 秒降至 0.478 秒。本次核对了 E2 发布清单中的全部 11 份文件摘要。这些时间来自上游实验，本机新增测量见前两节。[E1 审计](../evidence/e1/paired-audit.json) · [E2 成对重放](../evidence/e2/replay.json) · [E2 发布清单](../evidence/e2/validation.json)

下一步建议在同一份双层材料网格和冻结的通量、能量门限下，只改变预条件方式，逐项记录迭代次数、验证峰值内存和端到端时间。先确认两个故障对照仍能检出，再判断改善是否能在重复试验中稳定出现。

## 复现入口

在仓库说明指定的 Linux 环境中编译串行求解器后，以下命令可复现实验。每次测量使用新目录，BLAS 与 OpenMP 各限一个线程。完整性能命令还会对正弦任务运行重算与复用对照。MPI 的 9 项测试需要额外构建并行驱动。[串行构建](../README.md) · [MPI 构建](MPI_BUILD.md)

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q --require-mfem
RUNS_DIR=runs/review-demo bash scripts/run_demo.sh
python scripts/compare_baseline.py runs/review-demo
python -m benchmarks.performance --mixed-boundary --repeats 3 --output runs/review-mixed
python -m benchmarks.performance --repeats 3 --output runs/review-retry-and-assembly
python -m benchmarks.stage2 --directory runs/review-stage2 --report runs/review-stage2.json
python -m pytest -q -m mpi --require-mfem --require-mpi
```

随汇报交付的本地压缩包含装配试验、重试、规划和 MPI 的原始运行目录，以及本次日志。仓库保留逐次性能记录、回归结果、归档摘要和压缩包校验值。[文件与 SHA-256 索引](../evidence/stage-review/index.json)
