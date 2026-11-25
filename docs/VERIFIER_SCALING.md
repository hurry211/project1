# 独立验证成本与分层扩散实验记录

本轮从 `hurry211/project1` 的研究分支提交 `6a406ff` 出发，沿用串行双精度 MFEM 4.10、原 Poisson 算例、SciPy 独立参考求解和按文件校验的归档。我们用相同的导出场测量验证成本，并检验对齐材料界面的有限元结果能否同时满足场、通量和能量约束。[上一阶段实验](PERFORMANCE_STAGE.md) · [本轮实现与变更文件](https://github.com/hurry211/project1/commit/6a1335bb5be3a1c8380e6012cc43c6ac6405d7c5)

## 实验方法

**同一份输入上的验证器对照。** 旧版源码取自 `git archive 6a406ff`。每次为两版各启动一个新 Python 进程，从同一只读归档重新验证全部层级。每组各预热一次，正式试验逐对交替交换版本顺序，进程固定于同一逻辑 CPU，BLAS 线程数为 1。计时含启动、读取、积分、独立装配、SciPy 稀疏直接求解、输入哈希及输出。归档整体校验在计时区间外，在第一轮前和最后一轮后核对外部保存的清单摘要。峰值 RSS 取自每次验证进程的 Linux `VmHWM`，地址空间预算来自冻结策略。每一轮比较输入摘要、逐项检查的结论与阈值、决策、自由度和迭代数。数值比较覆盖每层 L²、H¹ 半范数误差和独立参考场相对差，采用相对容差 `1e-7`、绝对容差 `1e-12`。正式样本报告中位数、极值和中位绝对偏差（MAD）。[计时与一致性检查](../benchmarks/verifier_scaling.py)

**验证器改动。** 从原始顶点坐标和导出自由度坐标分别减去单元原点，逐项验证每个单元的局部几何及节点顺序是否完全相同。满足条件时复用局部基函数、雅可比、积分模板和常系数单元矩阵。节点排列或形状不同的单元继续逐个计算。误差积分分批执行，每批约六万五千个积分点。Neumann 项基于四个物理外边界寻找单元归属，边界标签被错换时仍能给出参考场差异。全局稀疏系统仍由验证器自行装配和直接求解，独立进程的时间、地址空间限额及归档重放均继续生效。[单元几何与基函数](../verifier/fem.py) · [独立积分与参考求解](../verifier/verifier.py)

**扩大载荷和规模。** 偏心高斯算例取 $r^2=(x-3/8)^2+(y-5/8)^2$，$u=e^{-25r^2}$，$f=(100-2500r^2)u$，四边施加解析 Dirichlet 数据。正式对照固定 Q2、CG + Jacobi、相对容差 `1e-12`，比较 MFEM 全装配与部分装配。两组网格分别为 [32,64,128] 和 [64,128,256]。每组各预热一次，再做三对交替顺序试验。原生各阶段耗时对三个网格层级求和，两类峰值内存分别取各自进程峰值。CG 迭代数与误差指最细层。端到端计时从调用研究流程开始，至写完最终清单，涵盖启动、导出、验证及归档。两种装配均须通过独立验收，各层误差还须等价。[冻结算例配置](../specs/experiments/gaussian_offset_q2.json) · [成对试验程序](../benchmarks/performance.py)

## 对齐界面的物理基准

单位方形在 $x=1/2$ 处分层，左侧 $a=1$，右侧 $a=10^4$。解满足 $-\nabla\cdot(a\nabla u)=0$，左右边界为 0 和 1，上下边界绝热。按 $a\nabla u$ 的记号，其水平分量为 $q=(1/2+1/(2\cdot10^4))^{-1}$。分段解为 $u=q[\min(x,1/2)+\max(x-1/2,0)/10^4]$，能量 $\int_\Omega a|\nabla u|^2=q$。研究分支的独立原型提供了该物理问题和场、通量、能量的容差依据。本轮把对比度固定为 10,000，并接入统一的算例注册、冻结输入、原生求解、独立验证和归档重放。[原型版本](https://github.com/hurry211/project1/blob/b7b4ec742a98c5303d0d67c2e03f8d064010a6a7/benchmarks/layered_diffusion.py)

对齐界面的 Q1 空间包含这个分段线性精确解。此时网格加密后的误差主要来自迭代和浮点计算，拟合渐近收敛阶没有稳定含义。冻结配置明确选择逐层物理验收，并要求独立参考求解及 CG 收敛。验证器按实际坐标恢复每个单元的四个角点，从 Q1 场重建两方向通量。界面同一高度的左右两侧逐段比较两个端点的法向通量，覆盖整个线性迹。体能量用线性函数平方的精确积分计算。节点最大绝对误差阈值为 `1e-8`，全域通量与界面跳跃相对于 $q$ 的阈值均为 `1e-5`，能量相对阈值为 `1e-7`。独立 SciPy 解使用按层加权的刚度矩阵。[算例与冻结准则](../specs/extensions/layered_unit_drop_10000.json) · [场、通量及能量计算](../verifier/layered.py)

## 实测结果

试验运行于 Ubuntu 24.04 WSL2，处理器为 AMD Ryzen 7 7840H，虚拟机可见 16 个逻辑 CPU 和约 7.35 GiB 内存。软件版本为 Python 3.12.3、GCC 13.3 和串行双精度 MFEM 4.10。正式性能试验固定在逻辑 CPU 2。机器可见 RTX 4060 Laptop GPU，本次 MFEM 构建的 MPI 和 CUDA 均未启用。下表时间以秒计，峰值常驻内存以 MiB 计。[硬件记录](../evidence/scaling-stage/hardware.txt) · [编译记录](../evidence/scaling-stage/build.log) · [环境、原生分项及单次样本](../evidence/scaling-stage/gaussian.json)

### 同一原始场上的验证成本

| 输入与最细自由度 | 正式配对数 | 旧版验证中位数 [最小, 最大] | 新版验证中位数 [最小, 最大] | 峰值 RSS 中位数，旧 → 新 | 跨版本等价结论 |
|---|---:|---:|---:|---:|---|
| 混合边界 Q2，66,049 | 5 | 4.084 [4.010, 4.475] | 1.980 [1.854, 2.175] | 527.7 → 336.6 | 5 对全部通过 |
| 偏心高斯 Q2，263,169 | 3 | 23.566 [23.421, 24.557] | 17.416 [17.386, 17.483] | 2,132.8 → 1,409.3 | 3 对均超出参考场诊断容差 |

混合边界组逐对取“旧版耗时 / 新版耗时”，加速比中位数为 **2.026 倍**，范围 [1.899, 2.339]，MAD 为 0.126。独立参考求解、归档核验和逐层数值检查均保留。这个加速比只计验证进程。[5 对重放及数值明细](../evidence/scaling-stage/verifier-mixed.json)

在最大高斯网格上，旧、新两版都接受同一归档。最细层的参考场相对差 $\|u_\mathrm{MFEM}-u_\mathrm{ref}\|_\infty/\|u_\mathrm{ref}\|_\infty$ 分别为 1.3342×10⁻¹² 和 4.9596×10⁻¹²，均低于算例的 10⁻⁷ 验收阈值。两个诊断值相差 3.6254×10⁻¹²，超过计时实验预设的 10⁻¹² 绝对等价阈值。该组只报告原始耗时和内存，加速比字段为 `null`。额外做了一次受同一预算约束的诊断重放，保留已核对的节点基函数，恢复逐单元物理几何的积分与装配，三个网格层级的参考场相对差均逐值复现旧版结果。这将偏差定位到了几何积分与局部装配的浮点计算路径。[3 对重放及预设阈值结果](../evidence/scaling-stage/verifier-gaussian.json) · [诊断程序](../evidence/scaling-stage/geometry_control.py) · [诊断输出](../evidence/scaling-stage/geometry-control.json)

### 偏心高斯载荷的装配与求解

| 最细自由度 | 装配 | 装配 ms | 预条件器 ms | CG s | 独立验证 s | 端到端 s [最小, 最大] | 原生 / 验证峰值 MiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 66,049 | 全 | 93.12 | 0.99 | 0.723 | 2.919 | 4.334 [4.198, 4.466] | 50.8 / 338.8 |
| 66,049 | 部分 | 23.09 | 1.73 | 1.627 | 2.931 | 5.218 [5.190, 5.307] | 30.7 / 338.5 |
| 263,169 | 全 | 427.84 | 3.28 | 8.922 | 17.791 | 29.032 [28.326, 29.405] | 178.5 / 1,408.2 |
| 263,169 | 部分 | 115.77 | 11.28 | 15.453 | 17.095 | 34.416 [34.152, 37.029] | 94.8 / 1,408.3 |

每个规模的三对试验均通过各层精度等价检查。两种装配在 66,049 个自由度上均需 849 次迭代，独立 L² 误差为 4.6807×10⁻⁷、H¹ 半范数误差为 3.8831×10⁻⁴。在 263,169 个自由度上均需 1,677 次迭代，两项误差分别为 5.8518×10⁻⁸ 和 9.7087×10⁻⁵。两个规模的“全装配 / 部分装配”端到端时间比中位数分别为 0.817 和 0.823，六个配对中全装配的端到端耗时都较短。部分装配明显降低了装配开销和原生峰值内存，其额外的迭代总耗时超过了装配节省的时间。[每次求解、误差、RSS、MAD 和摘要](../evidence/scaling-stage/gaussian.json)

### 分层扩散与策略选择

分层正例在 n=4、8、16、32、64、128 上逐层通过。下表列出 n=128 的四项物理检查，这一层有 16,641 个自由度，187 次 CG 迭代。双线性型能量为 1.999800019998，独立参考场相对差为 3.765×10⁻¹¹。[逐层数值和冻结阈值](../evidence/scaling-stage/layered-evidence.json)

| 物理检查 | 实测值 | 验收上限 |
|---|---:|---:|
| 节点最大绝对误差 | 3.593×10⁻¹¹ | 10⁻⁸ |
| 全域通量最大相对误差 | 5.817×10⁻⁸ | 10⁻⁵ |
| 界面法向通量跳跃相对值 | 1.513×10⁻⁸ | 10⁻⁵ |
| 能量相对误差 | 1.110×10⁻¹⁶ | 10⁻⁷ |

将材料换成均匀系数，或把零源项换成单位源项，MFEM 都正常结束。两组负例在 n=4、8、16 上均被独立验证拒绝。n=16 时，错误材料的界面通量跳跃相对值约为 5,000，错误源项为 0.03125。两组都触发了参考场、节点、通量与能量检查。移动归档后再用外部摘要重放，三个分层案例的结论均一致。[物理负例](../evidence/scaling-stage/layered-evidence.json) · [可移植归档重放](../evidence/scaling-stage/portable-replays.json)

在 66,049 个自由度的高斯算例上另跑了确定性策略搜索。固定 L² 上限 10⁻⁶、最多一次尝试、每个候选重复三次。能力矩阵排除尚未验证的两个 Q1 配置。两个 Q2 候选的六次运行全部通过，端到端中位数分别为全装配 4.042 秒、部分装配 5.011 秒，规则选中全装配。此次搜索总成本为 28.35 秒。[策略、候选、归档摘要与实测成本](../evidence/scaling-stage/gaussian-strategy.json)

### 回归与适用范围

原有 160 项测试保留，加入 10 项回归后共 170 项通过，包括真实 MFEM 集成测试。31 组历史任务的冻结输入、状态、尝试序列与原有故障码全部匹配。8 组通过验收，其他 23 组按历史预期拒绝、阻止执行或耗尽预算。已验收任务的 68 个误差范数与历史值的最大相对差为 5.28×10⁻¹²。验证器挑战中的 215 个故障样本均被拒绝，69 个干净样本均被接受，另有 14 个参考检查器校准样本。[测试日志](../evidence/scaling-stage/tests.log) · [31 组运行记录](../evidence/scaling-stage/demo.log) · [逐任务历史对照](../evidence/scaling-stage/legacy-comparison.json) · [挑战明细](../evidence/scaling-stage/challenge.json)

分层结论限于当前 1∶10,000、轴对齐界面、列出的六级 Q1 四边形网格、全装配和 CG + Gauss–Seidel。高斯装配结论限于本机串行 CPU、Q2 和已测网格。下一轮适合先评估分块逐单元参考装配能否保留大网格的诊断一致性，再用冻结的通量和能量门限比较强跳跃系数的预条件方法。MPI、GPU、非对齐界面与学习型选型仍待各自的对照实验。

## 复现

在 README 所列的 Ubuntu 24.04、锁定依赖和 MFEM 编译环境中，从研究分支根目录运行。输出实验应各使用一个新目录。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q --require-mfem
RUNS_DIR=runs/demo-replay bash scripts/run_demo.sh
python scripts/compare_baseline.py runs/demo-replay
python -m benchmarks.robustness --generate-controls runs/challenge-new --report runs/challenge-new.json
python -m sciagent run specs/extensions/layered_unit_drop_10000.json --runs-dir runs/layered
taskset -c 2 python -m benchmarks.performance --gaussian --repeats 3 --output runs/gaussian
```

同输入计时需要一份已验收的完整归档及其外部清单摘要。下面以第一阶段已公开清单摘要的混合边界归档为例。可用 CPU 编号因环境而异。

```bash
git worktree add --detach ../mfem-verifier-before 6a406ffae42d707df40177cbca19dea2dd8b5f3b
mkdir -p runs/replay-inputs
tar -xJf evidence/scaling-stage/replay-inputs.tar.xz -C runs/replay-inputs
taskset -c 2 python -m benchmarks.verifier_scaling \
  --baseline-root ../mfem-verifier-before \
  --attempt runs/replay-inputs/20261124T140228584587Z_memory-1024/attempts/attempt_01 \
  --manifest-sha256 52bbf0750bfa7b5b6b9271ff53b8ee829254f97c9c8c3d8aba2fcc8fb1258361 \
  --output runs/mixed-verifier.json --repeats 5
```

### 重放归档与最大网格诊断

仓库内 `replay-inputs.tar.xz` 含上一阶段的混合边界归档、本轮分层正例和两个分层负例。高斯最大网格的完整归档按三段保存，合并后 SHA-256 为 `10ded2be4ac13bd635c64385ea3d4d0abd131cdd30a3a32e861b776d2d7f7222`。五份原始运行目录均已移至新位置并用各自清单摘要重放，得到了三个预期接受和两个预期拒绝。[归档摘要与移动重放记录](../evidence/scaling-stage/portable-replays.json)

```bash
cat evidence/scaling-stage/gaussian-256.tar.xz.00[123] > runs/gaussian-256.tar.xz
echo '10ded2be4ac13bd635c64385ea3d4d0abd131cdd30a3a32e861b776d2d7f7222  runs/gaussian-256.tar.xz' | sha256sum -c -
tar -xJf runs/gaussian-256.tar.xz -C runs/replay-inputs
attempt=runs/replay-inputs/20261124T143653643465Z_q2-n256-full/attempts/attempt_01
pin=55c09ed6a3feb67d385a12767520326f4206c49626289e715e5a42553703cbcb
python -m sciagent verify "$attempt" --manifest-sha256 "$pin"
PYTHONPATH="$PWD" python evidence/scaling-stage/geometry_control.py "$attempt" "$pin"
```

用同一份高斯配置复现策略搜索时，固定 `validation.max_l2_error_finest=1e-6`、`policy.max_attempts=1` 和三级网格。按下列命令生成该实验输入。

```bash
python - <<'PY'
import json
from pathlib import Path

study = json.loads(Path("specs/experiments/gaussian_offset_q2.json").read_text())
study.update(validation={"max_l2_error_finest": 1e-6}, policy={"max_attempts": 1})
Path("runs/gaussian-strategy.json").write_text(json.dumps(study, indent=2))
PY
taskset -c 2 python -m sciagent search runs/gaussian-strategy.json --output runs/strategy-replay --repeats 3
```
