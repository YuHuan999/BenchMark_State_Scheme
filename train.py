import sys
sys.path.insert(0, r"E:\Projects\BenchMark_state_scheme\tianshou")

# import tianshou
# print("tianshou from:", tianshou.__file__)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from torch.distributions import Categorical
from qiskit import QuantumCircuit

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams

from env import CircuitDesignerDiscrete
from generate_task_suite.prepare_tasks import load_task_suite
from wrapper_represent import RepresentationWrapper
from encoders.encoder_MLP import Encoder_MLP

from typing import Dict, List, Tuple, Any

DEFAULT_BINS_ORDER = ["easy", "medium", "hard", "very_hard", "extreme"]

def normalize_tasks(tasks):
    # 你的 Prepare_tasks.py 用 target_qc；env 里建议你最终能读到 qc 或你在 env 里兼容 target_qc
    out = []
    for t in tasks:
        t = dict(t)
        if "qc" not in t and "target_qc" in t:
            t["qc"] = t["target_qc"]
        out.append(t)
    return out

def make_env(train_tasks, test_tasks, mode, scheme,
             seed=None, fidelity_threshold=0.99,
             success_reward=6.0, fail_reward=-5.0,
             gates_penalty=1.0, render_mode=None, task_pool=None):

    env = CircuitDesignerDiscrete(
        task_pool=task_pool,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        seed=seed,
        render_mode=render_mode,
        mode=mode,
        fidelity_threshold=fidelity_threshold,  
        success_reward=success_reward,
        fail_reward=fail_reward,
        gates_penalty=gates_penalty,
        # mode=mode  # 只有当你的 env __init__ 真有 mode 参数才传
    )
    env = RepresentationWrapper(env, scheme=scheme)
    return env

class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim=256, act="silu", use_ln=True):
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()

    def forward(self, z):
        return self.ln(self.act(self.proj(z)))

class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, shared: nn.Module, feat_dim: int, act_dim: int):
        super().__init__()
        self.encoder = encoder
        self.shared = shared

        self.actor_head = nn.Linear(feat_dim, act_dim)
        self.critic_head = nn.Linear(feat_dim, 1)

    def _feat(self, obs):
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(next(self.parameters()).device)

        z = self.encoder(state)   # [B, enc_dim]
        h = self.shared(z)        # [B, feat_dim]
        return h

    def forward_actor(self, obs, state=None, info=None):
        mask = obs.get("action_mask", None)
        feat = self._feat(obs)
        logits = self.actor_head(feat)

        if mask is not None:
            m = torch.as_tensor(mask, device=logits.device).bool()
            logits = logits.masked_fill(~m, -1e9)

        if info is not None and isinstance(info, dict):
            info["embedding"] = feat.detach()

        return logits, state

    def forward_critic(self, obs, state=None, info=None):
        feat = self._feat(obs)
        value = self.critic_head(feat).squeeze(-1)
        return value

class ActorWrapper(nn.Module):
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    def forward(self, obs, state=None, info=None):
        return self.ac.forward_actor(obs, state=state, info=info)

class CriticWrapper(nn.Module):
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    def forward(self, obs, state=None, info=None):
        return self.ac.forward_critic(obs, state=state, info=info)

def dist_fn(logits):
    return Categorical(logits=logits)

def sample_tasks(tasks: List[Dict[str, Any]],
                 bin_counts: List[int],
                 seed: int = 0,
                 bins_order: List[str] = DEFAULT_BINS_ORDER) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    按 difficulty_bin 分桶，并按 bin_counts 指定的数量（无放回）抽样。
    bin_counts 与 bins_order 一一对应，例如 [5,5,4,4,2] 对应 easy..extreme。
    """
    assert len(bin_counts) == len(bins_order), "bin_counts length must match bins_order"

    rng = np.random.default_rng(seed)

    # 1) 建桶（存 index）
    buckets = {b: [] for b in bins_order}
    for i, t in enumerate(tasks):
        b = t.get("difficulty_bin", None)
        if b is None:
            raise KeyError(f"Task missing difficulty_bin: task_id={t.get('task_id')}")
        b = str(b)
        if b not in buckets:
            raise ValueError(f"Unknown difficulty_bin={b} (task_id={t.get('task_id')})")
        buckets[b].append(i)

    # 2) 按 bin_counts 抽样（无放回，容量不足就报错）
    selected_idx = []
    for b, c in zip(bins_order, bin_counts):
        idxs = buckets[b]
        if c > len(idxs):
            raise ValueError(f"Not enough tasks in bin={b}: need {c}, have {len(idxs)}")
        chosen = rng.choice(idxs, size=c, replace=False).tolist()
        selected_idx.extend(chosen)

    selected_tasks = [tasks[i] for i in selected_idx]
    summary = {
        "seed": seed,
        "bins_order": bins_order,
        "bin_counts": dict(zip(bins_order, bin_counts)),
        "available_counts": {b: len(buckets[b]) for b in bins_order},
        "selected_total": len(selected_tasks),
    }
    return selected_tasks, summary

def run_one_task(task, *, scheme, net_cfg, algo_cfg, train_cfg, device, seed=0):
    """
    单任务从零训练 PPO，并在训练过程中定期评估，返回 metrics:
      solved, steps_to_solve, best_fidelity, final_fidelity, wall_time_sec
    """

    # -----------------------
    # 0) 固定随机性（尽量可复现）
    # -----------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()

    # -----------------------
    # 1) 固定为单任务池（关键）
    # -----------------------
    train_tasks = [task]
    test_tasks  = [task]

    n_train_env = int(train_cfg["n_train_env"])
    total_budget_steps = int(train_cfg["total_budget_steps"])
    eval_every_steps   = int(train_cfg["eval_every_steps"])   # 例如 10000
    eval_episodes      = int(train_cfg["eval_episodes"])      # 例如 1~3（Dev）/ 3~5（Val）
    buffer_size        = int(train_cfg["buffer_size"])

    # -----------------------
    # 2) 构建并行训练 env（同一个任务，不同 seed）
    # -----------------------
    fidelity_threshold_cfg = float(train_cfg.get("fidelity_threshold", 0.99))
    train_envs = DummyVectorEnv([
        (lambda i=i: (lambda: make_env(
            train_tasks, test_tasks,
            mode="train", scheme=scheme,
            seed=seed + i,
            fidelity_threshold=fidelity_threshold_cfg,
        )))()
        for i in range(n_train_env)
    ])

    # 单环境评估 env（同一个任务）
    eval_env = make_env(
        train_tasks, test_tasks,
        mode="test", scheme=scheme,
        seed=seed + 10_000,
        fidelity_threshold=fidelity_threshold_cfg,
    )

    # dummy env 推断维度
    dummy = make_env(
        train_tasks, test_tasks,
        mode="train", scheme=scheme,
        seed=seed + 999,
        fidelity_threshold=fidelity_threshold_cfg,
    )
    act_dim = int(dummy.action_space.n)
    max_gates = int(dummy.max_gates)

    actions = getattr(dummy, "actions", None) or getattr(dummy.unwrapped, "actions", None)
    if actions is None:
        from encoders.encoder_MLP import create_action_mapping
        actions = create_action_mapping(n_qubits=4)

    fidelity_threshold = float(getattr(dummy.unwrapped, "fidelity_threshold", 0.99))

    # -----------------------
    # 3) build network（只依赖 net_cfg）
    # -----------------------
    encoder = Encoder_MLP.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg).to(device)
    enc_out = int(getattr(encoder, "out_dim", net_cfg.get("hid", 256)))
    use_ln = bool(net_cfg.get("use_ln", True))
    shared_act = str(net_cfg.get("shared_act", "silu"))
    shared_out = int(net_cfg.get("shared_out_dim", enc_out))

    shared  = SharedMLP(in_dim=enc_out, out_dim=shared_out, act=shared_act, use_ln=use_ln).to(device)

    feat_dim = shared_out
    ac = ActorCritic(encoder, shared, feat_dim=feat_dim, act_dim=act_dim).to(device)
    actor  = ActorWrapper(ac)
    critic = CriticWrapper(ac)

    # -----------------------
    # 4) build PPO（只依赖 algo_cfg）
    # -----------------------
    lr = float(algo_cfg.get("lr", 3e-4))
    optim = AdamOptimizerFactory(lr=lr)

    actor_policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        action_space=dummy.action_space,
        observation_space=dummy.observation_space,
        action_scaling=False,
        action_bound_method=None,
    )

    algorithm = PPO(
        policy=actor_policy,
        critic=critic,
        optim=optim,
        gamma=float(algo_cfg.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
        max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        eps_clip=float(algo_cfg.get("eps_clip", 0.2)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        return_scaling=bool(algo_cfg.get("return_scaling", False)),
        advantage_normalization=bool(algo_cfg.get("advantage_normalization", True)),
        value_clip=bool(algo_cfg.get("value_clip", False)),
    )

    # -----------------------
    # 5) 评估函数：greedy(argmax)（稳定，适合比较超参）
    # -----------------------
    @torch.no_grad()
    def greedy_eval():
        actor.eval()
        succ = 0
        fids = []
        for ep in range(eval_episodes):
            obs, info = eval_env.reset(seed=seed + 20_000 + ep)
            done = False
            last_info = info
            while not done:
                obs_b = {
                    "state": np.expand_dims(obs["state"], axis=0),
                    "action_mask":  np.expand_dims(obs["action_mask"], axis=0) if "action_mask" in obs else None,
                }
                logits, _ = actor(obs_b)
                a = int(torch.argmax(logits[0]).item())
                obs, r, terminated, truncated, last_info = eval_env.step(a)
                done = bool(terminated or truncated)

            F = float(last_info.get("fidelity", 0.0))
            fids.append(F)
            if F >= fidelity_threshold:
                succ += 1
        return succ, float(np.mean(fids)) if fids else 0.0

    # -----------------------
    # 6) 训练循环：每 eval_every_steps 训练一次 + 评估一次
    #    => steps_to_solve 粒度就是 eval_every_steps
    # -----------------------
    collect_steps = int(train_cfg.get("collect_steps", 2048))
    update_reps   = int(train_cfg.get("update_reps", 6))
    batch_size    = int(train_cfg.get("batch_size", 256))

    solved = False
    steps_to_solve = None
    best_fidelity = -1.0
    final_fidelity = -1.0
    t_solve = None

    trained = 0
    while trained < total_budget_steps:
        this_chunk = min(eval_every_steps, total_budget_steps - trained)

        # 进度提示：架构/任务/seed + 已完成比例
        pct = 100.0 * trained / float(total_budget_steps)
        print(
            f"[train] cfg={net_cfg.get('name','net')} task={task.get('task_id')} "
            f"seed={seed} progress={trained}/{total_budget_steps} ({pct:.1f}%)"
        )

        # on-policy：每 chunk 用干净的 buffer（最简单可靠）
        buf = VectorReplayBuffer(total_size=max(buffer_size, this_chunk * 2), buffer_num=n_train_env)
        train_collector = Collector(algorithm, train_envs, buf, exploration_noise=False)

        trainer_params = OnPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=None,  # 我们不用 test_collector，自己 greedy_eval 更稳定
            max_epochs=1,
            epoch_num_steps=int(this_chunk),
            collection_step_num_env_steps=collect_steps,
            update_step_num_repetitions=update_reps,
            batch_size=batch_size,
            test_step_num_episodes=0,
            test_in_training=False,
        )

        algorithm.run_training(trainer_params)
        trained += this_chunk

        # 评估
        succ, mean_F = greedy_eval()
        final_fidelity = mean_F
        best_fidelity = max(best_fidelity, mean_F)

        if (not solved) and succ > 0:
            solved = True
            steps_to_solve = trained
            t_solve = time.time() - t0   # 第一次成功的耗时
            break  ## 暂时这样做，评估成功了就退出

    wall_time = time.time() - t0

    # 清理
    try:
        train_envs.close()
        eval_env.close()
    except Exception:
        pass

    return {
        "task_id": task.get("task_id", None),
        "difficulty_bin": task.get("difficulty_bin", None),
        "seed": int(seed),
        "budget_steps": int(total_budget_steps),
        "solved": int(solved),
        "steps_to_solve": (int(steps_to_solve) if steps_to_solve is not None else None),
        "best_fidelity": float(best_fidelity),
        "final_fidelity": float(final_fidelity),
        "wall_time_sec": float(wall_time),
        "time_to_solve_sec": float(t_solve) if t_solve is not None else wall_time * 10,
    }

def summarize_metrics(records, fidelity_threshold=0.99):
    """
    records: List[dict] from run_one_task(...)
    返回一个 summary dict，用于排名/打印。
    """
    if not records:
        return {
            "n_runs": 0,
            "success_rate": 0.0,
            "mean_steps_to_solve": None,
            "median_steps_to_solve": None,
            "mean_best_fidelity": 0.0,
            "mean_final_fidelity": 0.0,
            "mean_wall_time_sec": 0.0,
            "mean_time_to_solve_sec": 0.0,
        }

    solved_flags = np.array([r["solved"] for r in records], dtype=np.float32)
    success_rate = float(np.mean(solved_flags))

    steps = [r["steps_to_solve"] for r in records if r["steps_to_solve"] is not None]
    mean_steps = float(np.mean(steps)) if steps else None
    median_steps = float(np.median(steps)) if steps else None

    best_F = float(np.mean([r.get("best_fidelity", 0.0) for r in records]))
    final_F = float(np.mean([r.get("final_fidelity", 0.0) for r in records]))
    wall_t = float(np.mean([r.get("wall_time_sec", 0.0) for r in records]))
    time_to_solve_t = float(np.mean([r.get("time_to_solve_sec", 0.0) for r in records]))
    
    return {
        "n_runs": int(len(records)),
        "success_rate": success_rate,
        "mean_steps_to_solve": mean_steps,
        "median_steps_to_solve": median_steps,
        "mean_best_fidelity": best_F,
        "mean_final_fidelity": final_F,
        "mean_wall_time_sec": wall_t,
        "mean_time_to_solve_sec": time_to_solve_t,
    }

def run_cfg_on_taskset(
    *,
    stage_name: str,
    tasks: list,
    scheme: str,
    net_cfg: dict,
    algo_cfg: dict,
    train_cfg: dict,
    device: str,
    seeds: list,
    verbose: bool = True,
):
    """
    对一个网络结构 net_cfg，在 tasks 上、按 seeds 多次重复，跑完并汇总。
    """
    total_tasks = len(tasks)
    total_trials = total_tasks * len(seeds)
    trial_idx = 0

    all_records = []
    for ti, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{ti}")
        for s in seeds:
            trial_idx += 1
            trial_pct = 100.0 * trial_idx / float(total_trials)
            print(
                f"[run] cfg={net_cfg.get('name','net_cfg')} "
                f"task={task_id} ({ti+1}/{total_tasks}) "
                f"seed={s} trial={trial_idx}/{total_trials} ({trial_pct:.1f}%)"
            )

            rec = run_one_task(
                task,
                scheme=scheme,
                net_cfg=net_cfg,
                algo_cfg=algo_cfg,
                train_cfg=train_cfg,
                device=device,
                seed=int(s),
            )
            rec["stage"] = stage_name
            rec["net_cfg_name"] = net_cfg.get("name", "net_cfg")
            all_records.append(rec)

            if verbose:
                print(
                    f"[{stage_name}] cfg={rec['net_cfg_name']} "
                    f"task={task_id} bin={rec.get('difficulty_bin')} seed={s} "
                    f"solved={rec['solved']} steps_to_solve={rec['steps_to_solve']} "
                    f"bestF={rec['best_fidelity']:.4f} finalF={rec['final_fidelity']:.4f} "
                    f"time={rec['wall_time_sec']:.1f}s"
                )

    # 这里阈值如果你环境里不是 0.99，可以从 train_cfg 或 algo_cfg 里传
    summary = summarize_metrics(all_records, fidelity_threshold=train_cfg.get("fidelity_threshold", 0.99))
    summary["stage"] = stage_name
    summary["net_cfg_name"] = net_cfg.get("name", "net_cfg")
    return all_records, summary

def run_stage(
    *,
    in_dir: str,
    suite_name: str,          # "Dev"/"Val"/"Eval"
    scheme: str,
    bin_counts: list,         # e.g. [5,5,4,4,2]
    task_sample_seed: int,    # 固定抽样
    net_cfg_list: list,       # List[dict]
    algo_cfg: dict,
    train_cfg: dict,
    seeds: list,              # e.g. [0,1,2]
    device: str,
    verbose: bool = True,
):
    # 1) load tasks
    tasks = load_task_suite(in_dir, suite_name)
    tasks = normalize_tasks(tasks)

    # 2) sample tasks (可复现)
    sampled_tasks, sample_summary = sample_tasks(tasks, bin_counts=bin_counts, seed=task_sample_seed)

    if verbose:
        print(f"\n=== Stage {suite_name} / scheme={scheme} ===")
        print("Sample summary:", sample_summary)
        print("Sampled task_ids:", [t.get("task_id") for t in sampled_tasks])

    # 3) run each net cfg
    all_summaries = []
    all_records_by_cfg = {}

    for net_cfg in net_cfg_list:
        records, summary = run_cfg_on_taskset(
            stage_name=suite_name,
            tasks=sampled_tasks,
            scheme=scheme,
            net_cfg=net_cfg,
            algo_cfg=algo_cfg,
            train_cfg=train_cfg,
            device=device,
            seeds=seeds,
            verbose=verbose,
        )
        all_records_by_cfg[net_cfg.get("name", "net_cfg")] = records
        all_summaries.append(summary)

    # 4) ranking (默认：成功率优先，其次 steps_to_solve 越小越好，其次 best_fidelity 越大越好)
    def rank_key(s):
        # steps_to_solve None 说明没成功过 → 认为非常差
        steps = s["mean_steps_to_solve"]
        steps_key = steps if steps is not None else 1e18
        return (-s["success_rate"], steps_key, -s["mean_best_fidelity"], s["mean_wall_time_sec"])

    all_summaries_sorted = sorted(all_summaries, key=rank_key)

    if verbose:
        print("\n=== Ranking ===")
        for i, s in enumerate(all_summaries_sorted):
            print(
                f"{i+1:02d}. {s['net_cfg_name']}  "
                f"succ={s['success_rate']:.3f}  "
                f"mean_steps={s['mean_steps_to_solve']}  "
                f"bestF={s['mean_best_fidelity']:.4f}  "
                f"finalF={s['mean_final_fidelity']:.4f}  "
                f"time={s['mean_wall_time_sec']:.1f}s"
            )

    return {
        "sample_summary": sample_summary,
        "sampled_tasks": sampled_tasks,
        "summaries_sorted": all_summaries_sorted,
        "summaries_raw": all_summaries,
        "records_by_cfg": all_records_by_cfg,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Dev task 第一阶段
    in_dir = "task_suites"
    name = "Dev"  ## Dev, Eval

    dev_tasks = load_task_suite(in_dir, name)
    dev_tasks = normalize_tasks(dev_tasks)

    train_tasks = dev_tasks
    test_tasks  = dev_tasks
    scheme = "gate_seq"   # 你会用 argparse 切换
    n_train_env = 8
    n_test_env = 1
    base_seed = 1

    # 用 list comprehension，避免 [lambda]*n 的坑；并给不同 seed
    train_envs = DummyVectorEnv([
        (lambda i=i: (lambda: make_env(train_tasks, test_tasks, mode="train",
                                    scheme=scheme, seed=base_seed + i)))()
        for i in range(n_train_env)
    ])
    test_envs = DummyVectorEnv([
    (lambda i=i: (lambda: make_env(train_tasks, test_tasks, mode="test",
                                   scheme=scheme, seed=base_seed + 10_000 + i)))()
    for i in range(n_test_env)
    ])
    # 从 env 拿动作维度
    dummy = make_env(train_tasks, test_tasks, mode="train", scheme=scheme,
                     seed=base_seed + 999)
    act_dim = int(dummy.action_space.n)

    # ---- 从 env 推断 max_gates（gate_seq 表示一般就是 state 的长度）----
    obs0, _ = dummy.reset(seed=base_seed + 123)
    max_gates = int(obs0["state"].shape[-1])   # e.g. 32

    actions = getattr(dummy, "actions", None) or getattr(dummy.unwrapped, "actions", None)
    if actions is None:
        # 如果 encoder_MLP.py 里没有暴露 create_action_mapping，你就用你自己那份全局 actions
        # 这里假设你已经固定 Nmax=4
        from encoders.encoder_MLP import create_action_mapping
        actions = create_action_mapping(n_qubits=4)

    # ---- Encoder_MLP -> SharedMLP ----
    encoder = Encoder_MLP(actions=actions, max_gates=max_gates, hid=256).to(device)
    shared  = SharedMLP(in_dim=256, out_dim=256, act="silu", use_ln=True).to(device)

    feat_dim = 256
    ac = ActorCritic(encoder, shared, feat_dim=feat_dim, act_dim=act_dim).to(device)

    actor  = ActorWrapper(ac)
    critic = CriticWrapper(ac)
    optim = AdamOptimizerFactory(lr=3e-4)

    actor_policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        action_space=dummy.action_space,
        observation_space=dummy.observation_space,
        action_scaling=False,
        action_bound_method=None,
    )
    algorithm = PPO(
        policy=actor_policy,
        critic=critic,
        optim=optim,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01, # 促进探索（Dev 很重要）
        return_scaling=False,
        advantage_normalization=True, # 稳定训练（Dev 很重要）
        value_clip=False,
    )

    # on-policy buffer（大小 >= 每次收集的步数）
    buf = VectorReplayBuffer(total_size=20000, buffer_num=n_train_env)
    train_collector = Collector(algorithm, train_envs, buf, exploration_noise=False)
    test_collector  = Collector(algorithm, test_envs, exploration_noise=False)

    trainer_params = OnPolicyTrainerParams(
        training_collector=train_collector,
        test_collector=test_collector,

        # Dev 粗筛：总预算先缩小（每个 trial 快）
        max_epochs=8,
        epoch_num_steps=10000,               # 每个 epoch 的总交互步数（越大越慢）
        collection_step_num_env_steps=2048, # 每轮收集 rollout 步数（PPO 常用 1k~4k）

        # 更新强度：先别太大，避免过拟合/不稳定
        update_step_num_repetitions=6,      # 3~6 都行，Dev 我建议 5
        batch_size=256,                     # 256 OK（你现在也是 256）

        # Dev 测试：保留一点点信号就够了（不然 test 会占掉大量时间）
        test_step_num_episodes=5,           # 从 20 降到 5
        test_in_training=False,
    )
    result = algorithm.run_training(trainer_params)
    print(result)


if __name__ == "__main__":

    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 任务集
    in_dir = "task_suites"
    suite_name = "Dev"
    scheme = "gate_seq"

    # 抽样：你给的 bin_counts
    bin_counts = [5, 0, 0, 0, 0]
    task_sample_seed = 123

    # ----------------
    # 1) network configs (baseline grid)
    #   说明：net_cfg 通过 Encoder_MLP.from_cfg + SharedMLP 共同生效
    # ----------------
    net_cfg_list = [
        # 超小：先看 pipeline 能不能稳定跑通 + 是否能在 easy/medium 上有信号
        # {"name": "hid64_ln",   "hid": 64,  "use_ln": True,  "shared_act": "silu", "act": "silu"},
        # {"name": "hid64_noln", "hid": 64,  "use_ln": False, "shared_act": "silu", "act": "silu"},

        # 小：通常是“最小可用”baseline（建议重点看这个）
        {"name": "hid128_ln",   "hid": 128, "use_ln": True,  "shared_act": "silu", "act": "silu"},
        {"name": "hid128_noln", "hid": 128, "use_ln": False, "shared_act": "silu", "act": "silu"},

        # 中：如果 128 明显不够，再看 256
        # {"name": "hid256_ln",   "hid": 256, "use_ln": True,  "shared_act": "silu", "act": "silu"},
        # {"name": "hid256_noln", "hid": 256, "use_ln": False, "shared_act": "silu", "act": "silu"},
    ]

    # ----------------
    # 2) PPO algo cfg (两个版本：标准 / 更强探索)
    # ----------------
    algo_cfg = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "eps_clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.02,  # 比你原来的 0.01 更鼓励探索（Dev 阶段更有用）
        "advantage_normalization": True,
        "value_clip": False,
        "return_scaling": False,
    }

    # ----------------
    # 3) train cfg (Dev：先跑通 + 快速出信号)
    #   注意：run_one_task 里每 eval_every_steps 会重新建一次 on-policy buffer
    # ----------------
    train_cfg = {
        "n_train_env": 8,

        # 先别太大：调 bug / 看趋势用 30k~50k 更合适
        "total_budget_steps": 500000,

        # 每 5k 做一次 greedy_eval（steps_to_solve 粒度也就是 5k）
        "eval_every_steps": 5000,
        "eval_episodes": 1,

        # PPO rollout / update 强度：先保守，稳定优先
        "collect_steps": 1024,
        "update_reps": 4,
        "batch_size": 256,

        # buffer_size 只是下限，代码里会用 max(buffer_size, this_chunk*2) 兜底
        "buffer_size": 20000,

        # 需要的话你也能显式写死阈值（否则从 env 里读）
        "fidelity_threshold": 0.9,
    }

    # ----------------
    # 4) seeds（先 smoke：1 个；跑 baseline：2~3 个）
    # ----------------
    seeds = [0, 1, 2]


    # # 你要粗筛的网络结构列表（HPO 就是枚举/采样这些）
    # net_cfg_list = [
    #     {"name": "hid256_ln", "hid": 256, "use_ln": True,  "shared_act": "silu"},
    #     {"name": "hid256_noln", "hid": 256, "use_ln": False, "shared_act": "silu"},
    #     {"name": "hid128_ln", "hid": 128, "use_ln": True,  "shared_act": "silu"},
    # ]

    # # 算法超参（你可以先固定住）
    # algo_cfg = {
    #     "lr": 3e-4,
    #     "gamma": 0.99,
    #     "gae_lambda": 0.95,
    #     "max_grad_norm": 0.5,
    #     "eps_clip": 0.2,
    #     "vf_coef": 0.5,
    #     "ent_coef": 0.01,
    #     "advantage_normalization": True,
    #     "value_clip": False,
    # }

    # # 训练资源（预算）
    # train_cfg = {
    #     "n_train_env": 8,
    #     "total_budget_steps": 100000,
    #     "eval_every_steps": 10000,
    #     "eval_episodes": 1,
    #     "collect_steps": 2048,
    #     "update_reps": 6,
    #     "batch_size": 256,
    #     "buffer_size": 20000,
    #     # "fidelity_threshold": 0.99,  # 可选：你也可以从 env 读
    # }

    # # 多 seed（完成概率就靠它）
    # seeds = [0, 1, 2]

    out = run_stage(
        in_dir=in_dir,
        suite_name=suite_name,
        scheme=scheme,
        bin_counts=bin_counts,
        task_sample_seed=task_sample_seed,
        net_cfg_list=net_cfg_list,
        algo_cfg=algo_cfg,
        train_cfg=train_cfg,
        seeds=seeds,
        device=device,
        verbose=True,
    )

    # out["summaries_sorted"] 就是粗筛结果
    # out["records_by_cfg"] 里是所有明细（后续你可以保存成 jsonl/csv）


