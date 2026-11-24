# Stage 2 — 面向导师反馈的性能与极端问题研究

状态：开发分支候选实现；尚未获得本分支原生 MFEM 编译/运行结果，不宣称加速或新功能已验证。

## A. 深度性能优化

保持上游 MFEM 不变，新增实验入口 benchmarks.performance。全装配和部分装配使用同一数学模型、四边形网格、阶次、CG+Jacobi、停止阈值和独立验证阈值。全装配的 Q1/Q2 Jacobi 组合仍未在生产 capability 矩阵晋级，实验入口仅用于这两个候选组合。测试使用独立验证器，而非仅看 MFEM 退出码。

在 Ubuntu/WSL2 按 README 构建后执行（新输出目录）：

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ./scripts/build_solver.sh
    python -m pytest -q --require-mfem
    python -m benchmarks.performance --output runs/perf-q1 --order 1 --repeats 5 --levels 8 16 32
    python -m benchmarks.performance --output runs/perf-q2 --order 2 --repeats 5 --levels 8 16 32

报告包含每次实验原始值、每层中位数、端到端成对加速比和范围。计时区分 setup（含建网、装配）、assembly/system、CG配置与预条件器、迭代、误差计算、独立验证及端到端；部分时间段是包含关系，不能相加。Linux peak RSS 包括进程历史峰值而非装配增量，native 和 verifier 两个值分别记录，不相加作为同时峰值。验证器子进程有独立超时、地址空间限制；该隔离只应用于新实验入口，生产 controller 尚待改造。单次 CI smoke 仅验证路径，不足以形成稳定性能结论。

完整报告只有所有配对通过独立验证才标记 COMPLETE。保留 INCOMPLETE 和原始失败文件；不要用未完成或不同精度的结果估计加速。当前报告不是生产归档 manifest，不支持原生产 archive replay。后续扩展规模、预条件器及硬件前，先确认新增配置的 native integration test 成功，再审查生产能力矩阵。

## B. 特殊与极端问题功能扩展

首选强跳跃扩散（而非重写 MFEM）：
- 区域单位正方形，界面 x=1/2，a(x)=1（左）或 kappa（右），求解 -div(a grad u)=0。
- 左右边界分别 u=0 和 u=1；上下零法向通量。
- 精确通量 q=2*kappa/(kappa+1)，精确解左侧 q*x，右侧 q/2 + q*(x-1/2)/kappa。
- 界面条件为解连续、法向 a*grad u 连续；应检查界面两侧梯度、通量与能量积分（精确能量 q）。
- kappa=1, 10^2, 10^4, 10^6；网格必须对齐界面。Q1 已能表达分片线性精确解，因此不应套用光滑 Poisson 的 p+1 收敛阶判定。
- 后续构建独立可审查的 coefficient/problem adapter，加入错误系数、错界面、通量错误等负例；再测试非对齐界面、各向异性、奇异域及 AMG/MPI。禁止仅以残差或通用收敛阶接受这些问题。
- 最终研究目标是误差/通量/资源约束下的性能最优化，而不是盲目提高 MFEM 速度。记录搜索及验证开销、与固定策略及穷举基线对照。

## C. 下一步优先级

1. 原生 CI 通过，复核旧 31 组任务和验证器挑战未变。
2. 用同一机器多次测量，识别真正的时间、内存瓶颈；在生产工作流引入有资源上限的验证子进程及可靠的重试缓存（冻结输入、内容哈希、独立复核）。
3. 注册强跳跃扩散新算例，建立独立验证标准，完成 CPU 对照，再接入 MFEM 并行后端。
4. 基于经过验证的配置形成确定性自动选型基线；之后评估学习型/LLM 规划的实际增益。
