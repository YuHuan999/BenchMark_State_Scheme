# BenchMark_state_scheme （RL-QAS 量子电路合成基准）

## 项目概述
- 目标：比较不同 **state encoding scheme**（state representation + encoder）在量子电路合成/重建任务上的训练效率与成功率。
- 流程：对固定任务集（Dev/Val/Eval）抽样 → 为每个任务从零训练 PPO → 周期性 greedy 评估 → 汇总成功率、steps、fidelity、耗时。
- 关键特点：
  - 动作空间离散（最多 32），包含 4 量子比特上的 H/X/Y/Z/T/CNOT 组合。
  - 环境内置非法动作 mask，actor 前向强制 `masked_fill` 屏蔽非法动作。
  - 任务难度按 `difficulty_score = |A| * circuit_length` 分桶：easy/medium/hard/very_hard/extreme，可按桶数抽样复现。
  - 训练范式为“情况2”：每个任务独立从零训练，不强调跨任务泛化。

## 代码结构
- `train.py`：主入口，定义并调用以下模块：
  - `normalize_tasks` / `sample_tasks`：任务字段兼容、按桶抽样。
  - `make_env`：构建 `CircuitDesignerDiscrete` 并套 `RepresentationWrapper`。
  - `Encoder_MLP.from_cfg` + `SharedMLP` + `ActorCritic`：网络搭建。
  - `run_one_task`：单任务 PPO 训练 + 周期 greedy_eval，支持 early-stop（评估成功即 break）。
  - `run_cfg_on_taskset` / `run_stage`：多任务、多 seed、多网络配置的批量运行与汇总。
- `env.py`：量子电路环境（Gymnasium），核心：
  - 动作表 `_create_action_mapping`，非法动作 mask `_action_mask`（基于 frontier + 冗余/抵消规则）。
  - 观测为定长门序列（padding=-1），max_gates ≈ ceil(1.3 * 目标门数)。
  - 终止：达到保真度阈值 → terminated；或 budget 用尽 → truncated。
  - 奖励：`reward_cost`，成功给 `success_reward`，失败给 `fail_reward`，过程奖励基于 cost 改进。
- `wrapper_represent.py`：`RepresentationWrapper`，负责保持 `action_mask`，并按 `scheme` 转换观测（当前主要用 gate_seq）。
- `encoders/encoder_MLP.py`：可参数化的 MLP 编码器（含 `from_cfg`），gate_seq→onehot(含PAD)→MLP。
- `task_suites/`：任务集（Dev/Val/Eval），含元数据与电路文件。

## 运行依赖
- Python 3.11（示例解释器：`C:\Users\apple\anaconda3\envs\project_state_scheme311\python.exe`）
- 必需库：PyTorch、Gymnasium、Qiskit、NumPy、Tianshou（项目内自带 tianshou 源码并通过 `sys.path.insert` 引用）。

## 快速运行（当前推荐 smoke 配置）
1) 进入项目目录（Windows PowerShell）  
   ```powershell
   cd /d E:\Projects\BenchMark_state_scheme
   ```
2) 使用指定解释器运行：  
   ```powershell
   C:\Users\apple\anaconda3\envs\project_state_scheme311\python.exe  e:\Projects\BenchMark_state_scheme\train.py
   ```
3) 关键默认配置（`__main__`）：
   - 任务集：`task_suites/Dev`，仅 easy 桶抽样 `bin_counts=[5,0,0,0,0]`，`task_sample_seed=123`
   - 网络：`net_cfg_list = [hid128_ln, hid128_noln]`（可扩展 64/256 版本）
   - PPO 超参：lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, vf_coef=0.5, ent_coef=0.02, grad_norm=0.5
   - 训练：n_train_env=8，total_budget_steps=500000，eval_every_steps=5000，collect_steps=1024，update_reps=4，batch_size=256，fidelity_threshold=0.9
   - 种子：seeds=[0,1,2]
   - 早停：评估命中阈值即退出该任务训练，继续下一个任务/seed。

## 日志与进度
- 任务/seed 级进度：`[run] cfg=... task=... (ti/total_tasks) seed=s trial=idx/total_trials (pct%)`
- 单任务内步数进度：`[train] cfg=... task=... seed=... progress=trained/total_budget_steps (pct%)`
- 评估结果：`[Dev] cfg=... task=... bin=... seed=... solved=... steps_to_solve=... bestF=... finalF=... time=...s`
- 终端输出在 Cursor 的 `terminals/5.txt`（按当前后台运行的编号），可随时查看。

## 超参与可调项
- 抽样：`bin_counts`、`task_sample_seed` 控制任务难度与复现。
- 网络：`net_cfg` 支持 hid/out_dim/depth/act/use_ln/dropout/shared_out_dim；可扩展 256/更深 MLP。
- PPO：可调 ent_coef（探索）、collect_steps/update_reps（更新强度）、eval_every_steps（评估粒度）。
- 终止阈值：`fidelity_threshold`（train_cfg），或使用 env 默认值。
- 早停策略：`run_one_task` 内 `succ>0` 即 break；如需强制跑满预算，去掉 `break`。

## 常见问题与检查点
- **Mask 全链路**：env `_action_mask` → wrapper `obs["action_mask"]` → actor `masked_fill(~mask, -1e9)`；如报非法动作，先打印 mask 维度和值。
- **PAD 处理**：gate_seq padding=-1，Encoder_MLP 将 -1 映射到 pad 类（V=A+1）；如越界会抛异常。
- **步数上限**：`task_budget = ceil(1.3 * target_gates)`；达到预算会 truncated 结束。
- **阈值与奖励**：train_cfg.fidelity_threshold 会传给 env；成功奖励 `success_reward`，失败 `fail_reward`，过程奖励基于 cost 改进。

## 结果查看与导出
- 运行返回的 `out["records_by_cfg"]`、`out["summaries_sorted"]` 可在 `train.py` 里直接使用；需要持久化可添加保存为 jsonl/csv 的逻辑。
- 任务列表与难度分布可查看 `task_suites/Dev_meta.jsonl`；对应电路图在 `task_suites/circuit_png/Dev/`。

## 后续优化建议
- 先用更小预算/更少 seeds/net_cfg 做 smoke 验证（如 100k 步、1 seed、1~2 个 net_cfg），确认能命中阈值后再放大。
- 若始终 solved=0，可尝试：提高 ent_coef、增大 hid/深度、增大总步数，或降低阈值（已降到 0.9，可进一步调试）。
- 对 Val/Eval 阶段可恢复原始 bin_counts 分布，并开启更多网络/seed 进行对比。

