import sys
import os
import io
import json
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# 动态获取项目根目录，兼容Windows和Linux
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TIANSHOU_PATH = os.path.join(PROJECT_ROOT, "tianshou")
sys.path.insert(0, TIANSHOU_PATH)

# import tianshou
# print("tianshou from:", tianshou.__file__)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import subprocess
from collections import defaultdict
from qiskit import QuantumCircuit

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import policy_within_training_step
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams

from env import CircuitDesignerDiscrete
from generate_task_suite.prepare_tasks import load_task_suite
from wrapper_represent import RepresentationWrapper
from encoders.encoders_factory import build_encoder
from trace_recorder import infer_step, rollout_trace

from typing import Dict, List, Tuple, Any

DEFAULT_BINS_ORDER = ["easy", "medium", "hard", "very_hard", "extreme"]

# ============================================================================
# Scheme-Encoder 兼容性验证
# ============================================================================

# 定义 scheme（state representation）与 encoder 的兼容关系
SCHEME_ENCODER_COMPAT = {
    "gate_seq": ["mlp", "rnn"],          # 序列 token ids -> MLP 或 RNN
    "2d_grid": ["cnn"],                   # 2D grid -> CNN
    "3d_tensor": ["cnn"],                 # 3D tensor -> CNN
    "graph": ["gin"],                     # 图结构 -> GIN
}

def validate_scheme_encoder(scheme: str, encoder_type: str) -> None:
    """
    验证 scheme 和 encoder 是否兼容。
    
    Args:
        scheme: state representation 类型 ("gate_seq", "2d_grid", "3d_tensor", "graph")
        encoder_type: encoder 类型 ("mlp", "rnn", "cnn", "gin")
    
    Raises:
        ValueError: 如果 scheme 和 encoder 不兼容
    """
    encoder_type = (encoder_type or "mlp").lower()
    scheme = (scheme or "gate_seq").lower()
    
    valid_encoders = SCHEME_ENCODER_COMPAT.get(scheme, [])
    if not valid_encoders:
        raise ValueError(
            f"Unknown scheme: '{scheme}'. "
            f"Supported schemes: {list(SCHEME_ENCODER_COMPAT.keys())}"
        )
    
    if encoder_type not in valid_encoders:
        raise ValueError(
            f"scheme='{scheme}' is not compatible with encoder='{encoder_type}'. "
            f"Supported encoders for '{scheme}': {valid_encoders}"
        )


def init_lazy_encoder(encoder: nn.Module, dummy_env, encoder_type: str, device: str) -> int:
    """
    对于使用 LazyModule 的 encoder（GIN, CNN），用 dummy 数据初始化。
    
    Args:
        encoder: 已构建的 encoder 模块
        dummy_env: 用于获取 dummy observation 的 env
        encoder_type: encoder 类型
        device: 目标设备
    
    Returns:
        encoder_params: 初始化后的参数量
    """
    encoder_type = (encoder_type or "mlp").lower()
    
    if encoder_type in ("gin", "cnn"):
        # 获取一个真实的 observation 用于初始化
        obs, _ = dummy_env.reset()
        state = obs.get("state") if isinstance(obs, dict) else obs
        
        # 转换为 tensor 并 forward 一次以初始化 LazyModule
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.dim() == 1 or (state.dim() == 2 and encoder_type == "cnn"):
            state = state.unsqueeze(0)  # 添加 batch 维度
        state = state.to(device).float()
        
        with torch.no_grad():
            encoder(state)
    
    # 返回初始化后的参数量
    return encoder.num_params


def _try_get_optim_state(algorithm):
    for name in ["optim", "_optim", "optimizer", "_optimizer"]:
        opt = getattr(algorithm, name, None)
        if opt is not None and hasattr(opt, "state_dict"):
            try:
                return opt.state_dict()
            except Exception:
                continue
    return None


def _merge_trace_cache(actor_cache, value_cache, ac=None, buffer=None):
    """
    把 ActorCritic.trace_actor_cache / trace_value_cache 合并成 
    (z, h, logits, mask, value) numpy 数组。
    actor_cache: list of (z, h, logits, mask)，每项 shape=[B, ...]
    value_cache: list of value，每项 shape=[B]
    ac: ActorCritic 实例，用于在 value_cache 为空时通过 h 直接计算 value
    buffer: VectorReplayBuffer（未使用，保留接口兼容）
    返回：(z, h, logits, mask, value)，每项 shape=[total_steps, ...]
    
    注意：logits 会被 clip 到 float16 安全范围 [-65500, 65500]，避免 -inf 溢出。
    """
    if not actor_cache:
        return None, None, None, None, None

    # 先合并为 tensor
    z_cat = torch.cat([item[0] for item in actor_cache], dim=0)
    h_cat = torch.cat([item[1] for item in actor_cache], dim=0)
    logits_cat = torch.cat([item[2] for item in actor_cache], dim=0)
    mask_cat = torch.cat([item[3] for item in actor_cache], dim=0)
    
    z_all = z_cat.numpy()
    h_all = h_cat.numpy()
    logits_all = logits_cat.numpy()
    mask_all = mask_cat.numpy()
    
    # P0 修复：clip logits 到 float16 安全范围，避免 -inf 溢出
    # float16 范围是 [-65504, 65504]，留一点余量
    logits_all = np.clip(logits_all, -65500, 65500)

    if value_cache and len(value_cache) > 0:
        value_all = torch.cat(value_cache, dim=0).numpy()
    else:
        # P0 修复：如果 value_cache 为空（PPO collect 时 critic 不单独调用），
        # 直接用已有的 h（feat）通过 critic_head 计算 value，避免重新 forward encoder
        if ac is not None and hasattr(ac, "critic_head"):
            try:
                with torch.no_grad():
                    # h_cat 已经是 CPU tensor，需要移到正确的设备
                    device = next(ac.parameters()).device
                    h_device = h_cat.to(device)
                    value_tensor = ac.critic_head(h_device).squeeze(-1)
                    value_all = value_tensor.cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"[warn] Failed to compute value from h: {e}")
                value_all = np.zeros(z_all.shape[0], dtype=np.float32)
        else:
            # 没有 ac，填充 0
            value_all = np.zeros(z_all.shape[0], dtype=np.float32)

    return z_all, h_all, logits_all, mask_all, value_all


def _write_train_trace_from_buffer(buf, trace_data, out_path):
    """
    用 buffer 里的 (obs, act, rew, done, info) 和预先收集的 trace_data 写 trace 文件。
    trace_data: (z, h, logits, mask, value) numpy 数组，来自 _merge_trace_cache。
    如果 trace_data 为 None，则跳过 z/h/logits/mask/value 的写入（仅写 buffer 基础信息）。
    
    注意：VectorReplayBuffer 使用分段存储（每个环境一个子 buffer），
    必须用 sample_indices(0) 获取正确索引，而不是 np.arange(size)。
    
    重要：当 n_train_env > 1 时，trace_cache 的收集顺序（交错）与 buffer 存储顺序（分组）
    不一致。目前仅支持 n_train_env=1 的正确 trace 记录。
    """
    size = len(buf)
    if size == 0:
        return {"episodes": 0, "steps": 0, "path": out_path}

    # 修复：使用 sample_indices(0) 获取所有有效索引，而不是 np.arange(size)
    # 对于 VectorReplayBuffer，np.arange(size) 只会读取第一个子 buffer 的数据
    indices = buf.sample_indices(0)
    batch_all = buf[indices]
    
    # 检查是否是多环境（buffer_num > 1）
    buffer_num = getattr(buf, 'buffer_num', 1)
    if buffer_num > 1 and trace_data is not None:
        import warnings
        warnings.warn(
            f"n_train_env={buffer_num} > 1: trace_cache 顺序与 buffer 顺序不一致，"
            "trace 文件中的 z/h/logits/mask/value 可能与 action/reward 不匹配。"
            "建议使用 n_train_env=1 以确保 trace 数据正确性。",
            UserWarning
        )
    acts = batch_all.act
    rews = batch_all.rew
    dones = batch_all.done
    infos = batch_all.info
    # 修复：直接从 batch_all 读取 terminated 和 truncated，而不是从 info 中读取
    terms = batch_all.terminated
    truncs = batch_all.truncated

    # 解包 trace_data
    z_all, h_all, logits_all, mask_all, value_all = trace_data if trace_data else (None, None, None, None, None)

    env_ids = []
    ep_ids = []
    z_list = []
    h_list = []
    logits_list = []
    value_list = []
    mask_list = []
    act_list = []
    fid_after_list = []
    rew_after_list = []
    term_list = []
    trunc_list = []

    ep_counter = defaultdict(int)

    for i in range(size):
        # env_id / episode_id
        # 注意：info_i 可能是 dict 或 Batch 类型，两者都支持 .get() 方法
        info_i = infos[i]
        
        # 辅助函数：从 info_i 获取值（兼容 dict 和 Batch）
        def _get_info(key, default=None):
            if hasattr(info_i, 'get'):
                return info_i.get(key, default)
            elif hasattr(info_i, key):
                return getattr(info_i, key, default)
            return default
        
        env_id = int(_get_info("env_id", 0))
        env_ids.append(env_id)
        ep_ids.append(ep_counter[env_id])

        # 从 trace_data 取 z/h/logits/mask/value（不再二次前向）
        if z_all is not None and i < len(z_all):
            z_list.append(z_all[i].astype(np.float16))
            h_list.append(h_all[i].astype(np.float16))
            logits_list.append(logits_all[i].astype(np.float16))
            mask_list.append(mask_all[i].astype(np.uint8) if mask_all[i].dtype != np.uint8 else mask_all[i])
            value_list.append(np.float32(value_all[i]))
        else:
            # 没有 trace_data 时填充 placeholder（不应该发生）
            z_list.append(np.zeros(1, dtype=np.float16))
            h_list.append(np.zeros(1, dtype=np.float16))
            logits_list.append(np.zeros(1, dtype=np.float16))
            mask_list.append(np.zeros(1, dtype=np.uint8))
            value_list.append(np.float32(0.0))

        a = int(acts[i])
        done_flag = bool(dones[i])
        fid_after = float(_get_info("fidelity", 0.0))
        # 修复：直接使用 batch_all 中的 terminated 和 truncated
        terminated = bool(terms[i])
        truncated = bool(truncs[i])

        act_list.append(a)
        fid_after_list.append(fid_after)
        rew_after_list.append(float(rews[i]))
        term_list.append(np.uint8(terminated))
        trunc_list.append(np.uint8(truncated))

        if done_flag:
            ep_counter[env_id] += 1

    def _stack(lst, dtype=None):
        arr = np.stack(lst, axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    out = {
        "env_id": _stack(env_ids, np.int32),
        "episode_id": _stack(ep_ids, np.int32),
        "z": _stack(z_list, np.float16),
        "h": _stack(h_list, np.float16),
        "logits": _stack(logits_list, np.float16),
        "value": _stack(value_list, np.float32),
        "mask": _stack(mask_list, np.uint8),
        "action": _stack(act_list, np.int32),
        "fidelity_after": _stack(fid_after_list, np.float32),
        "reward_after": _stack(rew_after_list, np.float32),
        "terminated": _stack(term_list, np.uint8),
        "truncated": _stack(trunc_list, np.uint8),
    }
    np.savez_compressed(out_path, **out)
    return {"episodes": int(sum(term_list) + sum(trunc_list)), "steps": size, "path": out_path}


class RunLogger:
    """轻量级训练日志器：config/metrics/summary + TensorBoard."""

    def __init__(self, root_log_dir: str, scheme: str, stage_name: str, net_cfg_name: str, task_id: str, seed: int):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(
            root_log_dir,
            str(stage_name),
            str(scheme),
            str(net_cfg_name),
            str(task_id),
            f"seed_{seed}",
            f"run_{ts}",
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.run_dir, "tb")
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.writer = SummaryWriter(self.tb_dir)

    def _flatten(self, data: dict, prefix: str = ""):
        for k, v in data.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                yield from self._flatten(v, key)
            else:
                yield key, v

    def log_config(self, cfg: dict):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def log_metrics(self, step: int, metrics: dict):
        line = {"step": int(step), "timestamp": time.time()}
        line.update(metrics)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        for k, v in self._flatten(metrics):
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.writer.add_scalar(k, float(v), global_step=int(step))

    def log_summary(self, summary: dict):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def close(self):
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

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
             gates_penalty=1.0, render_mode=None, task_pool=None,
             max_gates=None):

    env_kwargs = dict(
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
    )
    # 允许从 train_cfg 显式控制观测长度；不传则沿用 env 的默认值
    if max_gates is not None:
        env_kwargs["max_gates"] = int(max_gates)

    env = CircuitDesignerDiscrete(**env_kwargs)
    env = RepresentationWrapper(env, scheme=scheme)
    return env

class SharedMLP(nn.Module):
    """
    Shared MLP layer between encoder and heads.
    
    Args:
        in_dim: 输入维度（encoder.out_dim）
        out_dim: 输出维度（默认 256）
        act: 激活函数 "silu" / "relu" / "gelu" / "none"（默认 "silu"）
        use_ln: 是否使用 LayerNorm（默认 True）
        dropout: Dropout 比率（默认 0.0，不使用）
    
    顺序：proj -> act -> ln -> dropout
    """
    def __init__(self, in_dim, out_dim=256, act="silu", use_ln=True, dropout=0.0):
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
        # 激活函数（支持 none）
        act_lower = (act or "none").lower()
        if act_lower == "silu" or act_lower == "swish":
            self.act = nn.SiLU()
        elif act_lower == "relu":
            self.act = nn.ReLU()
        elif act_lower == "gelu":
            self.act = nn.GELU()
        elif act_lower == "none" or act_lower == "identity":
            self.act = nn.Identity()
        else:
            self.act = nn.SiLU()  # 默认 silu
        
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, z):
        return self.dropout(self.ln(self.act(self.proj(z))))

class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, shared: nn.Module, feat_dim: int, act_dim: int):
        super().__init__()
        self.encoder = encoder
        self.shared = shared

        self.actor_head = nn.Linear(feat_dim, act_dim)
        self.critic_head = nn.Linear(feat_dim, 1)

        # ===== Trace 缓存：collect 时收集 z/h/logits/mask/value，不需二次前向 =====
        self.trace_enabled = False       # 是否启用 trace 缓存
        self.trace_actor_cache = []      # 每次 forward_actor 追加 (z, h, logits, mask)
        self.trace_value_cache = []      # 每次 forward_critic 追加 value

    def enable_trace(self):
        """启用 trace 缓存，collect 时调用。"""
        self.trace_enabled = True
        self.trace_actor_cache.clear()
        self.trace_value_cache.clear()

    def disable_trace(self):
        """禁用 trace 缓存并清空。"""
        self.trace_enabled = False
        self.trace_actor_cache.clear()
        self.trace_value_cache.clear()

    def get_trace_cache(self):
        """
        返回 (actor_cache, value_cache)，调用方负责合并。
        actor_cache: list of (z, h, logits, mask)  每项 shape=[B, ...]
        value_cache: list of value  每项 shape=[B]
        """
        return self.trace_actor_cache, self.trace_value_cache

    def _zh(self, obs):
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(next(self.parameters()).device)

        z = self.encoder(state)   # [B, enc_dim]
        h = self.shared(z)        # [B, feat_dim]
        return z, h

    def _feat(self, obs):
        return self._zh(obs)[1]

    def forward_actor(self, obs, state=None, info=None):
        mask = obs.get("action_mask", None)
        z, feat = self._zh(obs)
        logits = self.actor_head(feat)

        if mask is not None:
            m = torch.as_tensor(mask, device=logits.device).bool()
            logits = logits.masked_fill(~m, -1e9)

        # 写入外部传入的 info（用于 eval trace 等场景）
        if info is not None and isinstance(info, dict):
            info["z"] = z.detach()
            info["h"] = feat.detach()
            info["embedding"] = feat.detach()
            info["logits"] = logits.detach()
            # mask 也写入 info
            if mask is not None:
                info["mask"] = torch.as_tensor(mask, device=logits.device).detach()

        # ===== 如果启用 trace 缓存，追加到列表 =====
        if self.trace_enabled:
            mask_tensor = torch.as_tensor(mask, device=logits.device) if mask is not None else torch.ones_like(logits, dtype=torch.bool)
            self.trace_actor_cache.append((
                z.detach().cpu(),
                feat.detach().cpu(),
                logits.detach().cpu(),
                mask_tensor.detach().cpu(),
            ))

        return logits, state

    def forward_critic(self, obs, state=None, info=None):
        feat = self._feat(obs)
        value = self.critic_head(feat).squeeze(-1)

        # 写入外部传入的 info
        if info is not None and isinstance(info, dict):
            info["value"] = value.detach()

        # ===== 如果启用 trace 缓存，追加 value =====
        if self.trace_enabled:
            self.trace_value_cache.append(value.detach().cpu())

        return value

class ActorWrapper(nn.Module):
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac
        self.infer_time_total = 0.0
        self.infer_calls = 0

    def forward(self, obs, state=None, info=None):
        t0 = time.perf_counter()
        logits = self.ac.forward_actor(obs, state=state, info=info)
        self.infer_time_total += time.perf_counter() - t0
        self.infer_calls += 1
        return logits

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

def run_one_task(task, *, stage_name: str, scheme, net_cfg, algo_cfg, train_cfg, device, seed=0):
    """
    单任务从零训练 PPO，输出可复现的 run 目录与诊断指标。
    """

    # -----------------------
    # 0) 固定随机性（尽量可复现）
    # -----------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # -----------------------
    # 1) 固定为单任务池（关键）
    # -----------------------
    train_tasks = [task]
    test_tasks = [task]

    n_train_env = int(train_cfg["n_train_env"])
    total_budget_steps = int(train_cfg["total_budget_steps"])
    eval_every_steps = int(train_cfg["eval_every_steps"])   # 例如 10000
    eval_episodes = int(train_cfg["eval_episodes"])      # 例如 number of eval episodes
    buffer_size = int(train_cfg["buffer_size"])

    # -----------------------
    # 2) 构建并行训练 env（同一个任务，不同 seed）
    # -----------------------
    fidelity_threshold_cfg = float(train_cfg.get("fidelity_threshold", 0.99))
    # ⚠️ 正式实验时建议设置 early_stop_on_success=False，避免影响电路质量优化
    early_stop_on_success = bool(train_cfg.get("early_stop_on_success", True))
    early_stop_consecutive = int(train_cfg.get("early_stop_consecutive_success", 1))
    early_stop_consecutive = max(1, early_stop_consecutive)
    max_gates_cfg = train_cfg.get("max_gates", None)
    train_envs = DummyVectorEnv([
        (lambda i=i: (lambda: make_env(
            train_tasks, test_tasks,
            mode="train", scheme=scheme,
            seed=seed + i,
            fidelity_threshold=fidelity_threshold_cfg,
            max_gates=max_gates_cfg,
        )))()
        for i in range(n_train_env)
    ])

    # 单环境评估 env（同一个任务）
    eval_env = make_env(
        train_tasks, test_tasks,
        mode="test", scheme=scheme,
        seed=seed + 10_000,
        fidelity_threshold=fidelity_threshold_cfg,
        max_gates=max_gates_cfg,
    )

    # dummy env 推断维度
    dummy = make_env(
        train_tasks, test_tasks,
        mode="train", scheme=scheme,
        seed=seed + 999,
        fidelity_threshold=fidelity_threshold_cfg,
        max_gates=max_gates_cfg,
    )
    act_dim = int(dummy.action_space.n)
    max_gates = int(dummy.max_gates)

    actions = getattr(dummy, "actions", None) or getattr(dummy.unwrapped, "actions", None)
    if actions is None:
        raise RuntimeError("env must provide actions mapping (dummy.actions or dummy.unwrapped.actions)")

    fidelity_threshold = float(getattr(dummy.unwrapped, "fidelity_threshold", 0.99))

    # -----------------------
    # 3) build network（只依赖 net_cfg）
    # -----------------------
    # 获取 encoder 类型（默认 mlp）
    encoder_type = (
        net_cfg.get("encoder") or
        net_cfg.get("enc_name") or
        net_cfg.get("enc_type") or
        "mlp"
    ).lower()
    
    # 验证 scheme 和 encoder 的兼容性
    validate_scheme_encoder(scheme, encoder_type)
    
    # 构建 encoder
    encoder = build_encoder(actions=actions, max_gates=max_gates, net_cfg=net_cfg).to(device)
    
    # 对于使用 LazyModule 的 encoder（GIN, CNN），用 dummy 数据初始化
    encoder_params = init_lazy_encoder(encoder, dummy, encoder_type, device)
    
    enc_out = int(getattr(encoder, "out_dim", net_cfg.get("out_dim", 256)))
    use_ln = bool(net_cfg.get("use_ln", True))
    shared_act = str(net_cfg.get("shared_act", "silu"))
    shared_out = int(net_cfg.get("shared_out_dim", enc_out))
    shared_dropout = float(net_cfg.get("shared_dropout", 0.0))

    shared = SharedMLP(in_dim=enc_out, out_dim=shared_out, act=shared_act, use_ln=use_ln, dropout=shared_dropout).to(device)

    feat_dim = shared_out
    ac = ActorCritic(encoder, shared, feat_dim=feat_dim, act_dim=act_dim).to(device)
    actor = ActorWrapper(ac)
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
    
    algorithm.profile_update_time = 0.0
    algorithm.profile_update_calls = 0
    algorithm.profile_ppo_stats_sum = defaultdict(float)
    algorithm.profile_ppo_stats_n = 0
    orig_update = algorithm.update

    def _acc_stat(res_dict, aliases, key):
        for k in aliases:
            if k in res_dict and res_dict[k] is not None:
                try:
                    v = float(res_dict[k])
                    algorithm.profile_ppo_stats_sum[key] += v
                except Exception:
                    pass
                break

    def wrapped_update(*args, **kwargs):
        """
        P1 修复：正确捕获 PPO 训练统计信息。
        Tianshou PPO.update() 返回 TrainingStats 对象，需要调用 get_loss_stats_dict() 获取 dict。
        """
        t_start = time.perf_counter()
        res = orig_update(*args, **kwargs)
        algorithm.profile_update_time += time.perf_counter() - t_start
        algorithm.profile_update_calls += 1

        res_dict = None
        if isinstance(res, dict):
            res_dict = res
        elif hasattr(res, "get_loss_stats_dict"):
            # P1 修复：Tianshou TrainingStats 使用 get_loss_stats_dict() 方法
            try:
                res_dict = res.get_loss_stats_dict()
            except Exception:
                pass
        
        # 备选：尝试直接访问属性（适用于旧版本 Tianshou）
        if res_dict is None and hasattr(res, "__dict__"):
            res_dict = res.__dict__

        if res_dict is not None and isinstance(res_dict, dict) and len(res_dict) > 0:
            algorithm.profile_ppo_stats_n += 1
            # 更全面的 key 别名，适配 Tianshou 不同版本
            _acc_stat(res_dict, ["approx_kl", "kl", "policy_kl", "train/approx_kl", "loss/kl"], "approx_kl")
            _acc_stat(res_dict, ["clipfrac", "clip_frac", "policy_clipfrac", "clip_fraction", "loss/clip"], "clipfrac")
            _acc_stat(res_dict, ["entropy", "ent", "policy_entropy", "ent_loss", "loss/ent"], "entropy")
            _acc_stat(res_dict, ["loss", "total_loss", "loss/total"], "loss_total")
            _acc_stat(res_dict, ["actor_loss", "pi_loss", "loss/actor", "policy_loss"], "loss_actor")
            _acc_stat(res_dict, ["critic_loss", "vf_loss", "loss/critic", "value_loss", "loss/vf"], "loss_critic")
        return res

    algorithm.update = wrapped_update

    # -----------------------
    # 5) 日志器（每个 run 写 config/metrics/summary + TB）
    # -----------------------
    log_root = train_cfg.get("log_dir", "logs")
    task_id = task.get("task_id", None) or task.get("id", "task")
    logger = RunLogger(
        root_log_dir=log_root,
        scheme=scheme,
        stage_name=stage_name,
        net_cfg_name=net_cfg.get("name", "net_cfg"),
        task_id=task_id,
        seed=seed,
    )
    # trace / ckpt 目录
    traces_train_dir = os.path.join(logger.run_dir, "traces", "train")
    traces_eval_dir = os.path.join(logger.run_dir, "traces", "eval")
    # ckpt_dir = os.path.join(logger.run_dir, "checkpoints")  # 暂时禁用 checkpoint 功能
    os.makedirs(traces_train_dir, exist_ok=True)
    os.makedirs(traces_eval_dir, exist_ok=True)
    # os.makedirs(ckpt_dir, exist_ok=True)  # 暂时禁用 checkpoint 功能
    chunks_index_path = os.path.join(logger.run_dir, "chunks.jsonl")
    if not os.path.exists(chunks_index_path):
        open(chunks_index_path, "a", encoding="utf-8").close()

    # 记录 config 快照（一次性）
    config_snapshot = {
        "run_id": run_id,
        "stage": stage_name, ## suite name
        "scheme": scheme,
        "encoder_type": encoder_type,          # 新增：encoder 类型
        "encoder_params": encoder_params,      # 新增：encoder 参数量
        "task_id": task.get("task_id", None),
        "difficulty_bin": task.get("difficulty_bin", None),
        "seed": int(seed),
        "net_cfg": net_cfg,
        "algo_cfg": algo_cfg,
        "train_cfg": train_cfg,
        "device": device,
        "act_dim": act_dim,
        "max_gates": max_gates,
        "fidelity_threshold": fidelity_threshold,
        "actions": actions,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "tianshou_module_path": getattr(sys.modules.get(PPO.__module__, None), "__file__", None),
        "git_commit": None,
        "seed_everything": int(seed),
        "trace_schema": [
            "env_id", "episode_id", "z", "h", "logits", "value", "mask",
            "action", "fidelity_after", "reward_after", "terminated", "truncated"
        ],
        "trace_record_phase": "pre_action",
        "trace_logits_are_masked": True,
    }
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
        config_snapshot["git_commit"] = git_commit
    except Exception:
        pass
    logger.log_config(config_snapshot)  # 配置/指纹快照

    # -----------------------
    # 6) 评估函数：greedy(argmax)（稳定，适合比较超参）
    # -----------------------
    @torch.no_grad()
    def greedy_eval(eval_trace_path=None):
        succ = 0
        fids = []
        best_qasm = None
        best_stats = None
        mask_ratios = []
        illegal_cnt = 0
        total_steps = 0
        ep_times = []
        ep_steps = []
        entropies = []
        p_max_list = []
        margin_list = []
        k_eff_list = []

        trace_env_ids = []
        trace_ep_ids = []
        trace_z = []
        trace_h = []
        trace_logits = []
        trace_value = []
        trace_mask = []
        trace_action = []
        trace_fid_after = []
        trace_rew_after = []
        trace_term = []
        trace_trunc = []

        for ep in range(eval_episodes):
            ep_t0 = time.perf_counter()
            obs, info = eval_env.reset(seed=seed + 20_000 + ep)
            done = False
            last_info = info
            while not done:
                z_np, h_np, logits_np, value_np, mask_np = infer_step(ac, obs, device)
                probs = torch.softmax(torch.from_numpy(logits_np), dim=-1)
                p_sorted = torch.sort(probs, descending=True).values
                p_max = float(p_sorted[0].item())
                p_max_list.append(p_max)
                if p_sorted.numel() > 1:
                    margin = float((p_sorted[0] - p_sorted[1]).item())
                else:
                    margin = 0.0
                margin_list.append(margin)
                entropy = float(Categorical(logits=torch.from_numpy(logits_np)).entropy().item())
                entropies.append(entropy)
                k_eff = float(1.0 / max(1e-12, torch.sum(probs * probs).item()))
                k_eff_list.append(k_eff)

                a = int(np.argmax(logits_np))
                obs, r, terminated, truncated, last_info = eval_env.step(a)
                done = bool(terminated or truncated)
                mask_eval = last_info.get("action_mask", None) if isinstance(last_info, dict) else None
                if mask_eval is not None:
                    mask_ratios.append(float(np.mean(mask_eval)))
                if isinstance(last_info, dict) and last_info.get("illegal", False):
                    illegal_cnt += 1
                total_steps += 1

                trace_env_ids.append(0)
                trace_ep_ids.append(ep)
                trace_z.append(z_np.astype(np.float16))
                trace_h.append(h_np.astype(np.float16))
                # P0 修复：clip logits 到 float16 安全范围，避免 -inf 溢出
                trace_logits.append(np.clip(logits_np, -65500, 65500).astype(np.float16))
                trace_value.append(value_np)
                trace_mask.append(mask_np.astype(np.uint8))
                trace_action.append(a)
                trace_fid_after.append(float(last_info.get("fidelity", 0.0)))
                trace_rew_after.append(float(r))
                trace_term.append(np.uint8(terminated))
                trace_trunc.append(np.uint8(truncated))

            ep_steps.append(int(last_info.get("step_count", len(getattr(eval_env, "gate_tokens", [])))))
            ep_times.append(time.perf_counter() - ep_t0)

            F = float(last_info.get("fidelity", 0.0))
            fids.append(F)
            if F >= fidelity_threshold:
                succ += 1
                # 保留成功时的电路（qasm）
                try:
                    best_qasm = eval_env._qc.qasm()  # noqa: SLF001
                except Exception:
                    best_qasm = None
                try:
                    best_stats = eval_env.get_circuit_stats()
                except Exception:
                    best_stats = None
        mask_ratio_mean = float(np.mean(mask_ratios)) if mask_ratios else 0.0
        illegal_rate = float(illegal_cnt) / max(1, total_steps)
        mean_ep_time = float(np.mean(ep_times)) if ep_times else 0.0
        mean_ep_steps = float(np.mean(ep_steps)) if ep_steps else 0.0
        entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        p_max_mean = float(np.mean(p_max_list)) if p_max_list else 0.0
        margin_mean = float(np.mean(margin_list)) if margin_list else 0.0
        k_eff_mean = float(np.mean(k_eff_list)) if k_eff_list else 0.0

        if eval_trace_path is not None and trace_action:
            def _stack(lst, dtype=None):
                arr = np.stack(lst, axis=0)
                return arr.astype(dtype) if dtype is not None else arr
            np.savez_compressed(
                eval_trace_path,
                env_id=_stack(trace_env_ids, np.int32),
                episode_id=_stack(trace_ep_ids, np.int32),
                z=_stack(trace_z, np.float16),
                h=_stack(trace_h, np.float16),
                logits=_stack(trace_logits, np.float16),
                value=_stack(trace_value, np.float32),
                mask=_stack(trace_mask, np.uint8),
                action=_stack(trace_action, np.int32),
                fidelity_after=_stack(trace_fid_after, np.float32),
                reward_after=_stack(trace_rew_after, np.float32),
                terminated=_stack(trace_term, np.uint8),
                truncated=_stack(trace_trunc, np.uint8),
            )

        return (
            succ,
            float(np.mean(fids)) if fids else 0.0,
            best_qasm,
            best_stats,
            mask_ratio_mean,
            illegal_rate,
            mean_ep_time,
            mean_ep_steps,
            entropy_mean,
            p_max_mean,
            margin_mean,
            k_eff_mean,
        )

    # -----------------------
    # 7) 训练循环：每 eval_every_steps 训练一次 + 评估一次
    #    => steps_to_solve 粒度就是 eval_every_steps
    # -----------------------
    collect_steps = int(train_cfg.get("collect_steps", 2048))
    update_reps = int(train_cfg.get("update_reps", 6))
    batch_size = int(train_cfg.get("batch_size", 256))

    solved = False
    steps_to_solve = None
    best_fidelity = -1.0
    final_fidelity = -1.0
    best_stats_global = None
    best_qasm_global = None
    # 追踪达到目标的最短电路（按 gate_count 最小）
    min_stats_global = None
    min_qasm_global = None
    t_solve = None  # 记录首次成功（解出目标）的耗时
    solved_qasm = None
    # 记录目标电路的 qasm 便于对比
    target_qasm = None
    try:
        target_qc = task.get("qc") or task.get("target_qc")
        if target_qc is not None:
            target_qasm = target_qc.qasm()
    except Exception:
        target_qasm = None

    trained = 0
    chunk_idx = 0
    consecutive_successes = 0  # 连续成功的评估次数

    # profile 累积器（用差分得到"本 chunk"均值）
    def _env_profile_snapshot():
        """
        P1 修复：使用 train_envs.get_env_attr("profile") 正确获取所有子环境的 profile。
        DummyVectorEnv 不直接暴露 envs 属性，需要使用 get_env_attr 方法。
        """
        stats = dict(env_step_time=0.0, env_step_calls=0, illegal_count=0, mask_valid_sum=0.0, mask_calls=0)
        try:
            # 使用 Tianshou VectorEnv 的正确 API 获取所有子环境的 profile
            profiles = train_envs.get_env_attr("profile")  # 返回 list[dict]
            for prof in profiles:
                if not prof or not isinstance(prof, dict):
                    continue
                stats["env_step_time"] += float(prof.get("env_step_time", 0.0))
                stats["env_step_calls"] += int(prof.get("env_step_calls", 0))
                stats["illegal_count"] += int(prof.get("illegal_count", 0))
                stats["mask_valid_sum"] += float(prof.get("mask_valid_sum", 0.0))
                stats["mask_calls"] += int(prof.get("mask_calls", 0))
        except Exception as e:
            # 如果获取失败，返回默认值
            pass
        return stats

    prev_env_prof = _env_profile_snapshot()
    prev_infer_time = actor.infer_time_total
    prev_infer_calls = actor.infer_calls
    prev_update_time = algorithm.profile_update_time
    prev_update_calls = algorithm.profile_update_calls
    prev_ppo_n = algorithm.profile_ppo_stats_n
    prev_ppo_sum = defaultdict(float)
    mean_episode_time = 0.0
    last_entropy_mean = 0.0
    last_p_max_mean = 0.0
    last_margin_mean = 0.0
    last_k_eff_mean = 0.0

    try:
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
            train_collector.reset_env()

            # ===== 启用 trace 缓存：collect 时自动记录 z/h/logits/mask/value =====
            ac.enable_trace()

            # 1) 采样 this_chunk 步，使用当前策略（未更新前）
            t_train_start = time.perf_counter()
            train_collector.collect(n_step=int(this_chunk))
            train_chunk_time = time.perf_counter() - t_train_start
            trained += this_chunk

            # ===== 禁用 trace 并取出缓存数据 =====
            actor_cache, value_cache = ac.get_trace_cache()
            # P0 修复：传入 ac 和 buffer，以便在 value_cache 为空时重新计算 value
            trace_data = _merge_trace_cache(actor_cache, value_cache, ac=ac, buffer=train_collector.buffer)
            ac.disable_trace()

            # 2) 训练 trace（直接用缓存数据，不再二次前向）
            train_trace_path = os.path.join(
                traces_train_dir, f"chunk_{chunk_idx:04d}_step_{trained:08d}.npz"
            )
            _write_train_trace_from_buffer(train_collector.buffer, trace_data, train_trace_path)

            # 3) PPO 更新（重复 update_reps 次）
            for _ in range(update_reps):
                with policy_within_training_step(algorithm.policy):
                    algorithm.update(train_collector.buffer, batch_size=batch_size, repeat=1)

            # on-policy：更新后清空 buffer
            train_collector.reset_buffer()

            # 评估
            t_eval_start = time.perf_counter()
            eval_trace_path = os.path.join(
                traces_eval_dir, f"chunk_{chunk_idx:04d}_step_{trained:08d}.npz"
            )
            (
                succ,
                mean_F,
                solved_qasm_candidate,
                best_stats_candidate,
                mask_ratio_mean,
                illegal_rate_eval,
                mean_ep_time_eval,
                mean_ep_steps_eval,
                entropy_mean_eval,
                p_max_mean_eval,
                margin_mean_eval,
                k_eff_mean_eval,
            ) = greedy_eval(eval_trace_path=eval_trace_path)
            eval_time = time.perf_counter() - t_eval_start
            final_fidelity = mean_F
            best_fidelity = max(best_fidelity, mean_F)
            mean_episode_time = mean_ep_time_eval
            last_entropy_mean = entropy_mean_eval
            last_p_max_mean = p_max_mean_eval
            last_margin_mean = margin_mean_eval
            last_k_eff_mean = k_eff_mean_eval

            # # 保存 checkpoint（模型/优化器/RNG）—— 暂时禁用
            # ckpt_path = os.path.join(
            #     ckpt_dir, f"ckpt_chunk_{chunk_idx:04d}_step_{trained:08d}.pt"
            # )
            # optim_state = _try_get_optim_state(algorithm)
            # ckpt = {
            #     "chunk_idx": chunk_idx,
            #     "trained_steps": trained,
            #     "ac_state_dict": ac.state_dict(),
            #     "algorithm_state_dict": algorithm.state_dict(),
            #     "optim_state_dict": optim_state,
            #     "rng": {
            #         "numpy": np.random.get_state(),
            #         "torch": torch.get_rng_state(),
            #         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            #     },
            # }
            # torch.save(ckpt, ckpt_path)

            if (not solved) and succ > 0:
                solved = True
                steps_to_solve = trained
                t_solve = time.time() - t0   # 第一次成功的耗时
                if solved_qasm is None:
                    solved_qasm = solved_qasm_candidate
            # 连续成功计数（用于更稳健的提前停止）
            if succ > 0:
                consecutive_successes += 1
            else:
                consecutive_successes = 0

            if succ > 0 and best_stats_candidate is not None:
                best_stats_global = best_stats_candidate
                best_qasm_global = solved_qasm_candidate
                # 更新最短电路：只有门数更少时才更新
                current_gc = best_stats_candidate.get("gate_count", float('inf'))
                min_gc = (min_stats_global or {}).get("gate_count", float('inf'))
                if current_gc < min_gc:
                    min_stats_global = best_stats_candidate
                    min_qasm_global = solved_qasm_candidate

            # profile 增量
            env_prof = _env_profile_snapshot()
            delta_env_step_time = env_prof["env_step_time"] - prev_env_prof["env_step_time"]
            delta_env_calls = env_prof["env_step_calls"] - prev_env_prof["env_step_calls"]
            delta_illegal = env_prof["illegal_count"] - prev_env_prof["illegal_count"]
            delta_mask_sum = env_prof["mask_valid_sum"] - prev_env_prof["mask_valid_sum"]
            delta_mask_calls = env_prof["mask_calls"] - prev_env_prof["mask_calls"]
            prev_env_prof = env_prof

            delta_infer_time = actor.infer_time_total - prev_infer_time
            delta_infer_calls = actor.infer_calls - prev_infer_calls
            prev_infer_time = actor.infer_time_total
            prev_infer_calls = actor.infer_calls

            delta_update_time = algorithm.profile_update_time - prev_update_time
            delta_update_calls = algorithm.profile_update_calls - prev_update_calls
            prev_update_time = algorithm.profile_update_time
            prev_update_calls = algorithm.profile_update_calls
            delta_ppo_n = algorithm.profile_ppo_stats_n - prev_ppo_n
            ppo_means = {}
            if delta_ppo_n > 0:
                for k, v in algorithm.profile_ppo_stats_sum.items():
                    mean_v = (v - prev_ppo_sum.get(k, 0.0)) / delta_ppo_n
                    ppo_means[k] = mean_v
                prev_ppo_n = algorithm.profile_ppo_stats_n
                prev_ppo_sum = algorithm.profile_ppo_stats_sum.copy()

            steps_per_sec = this_chunk / train_chunk_time if train_chunk_time > 0 else 0.0
            env_step_time_avg = delta_env_step_time / max(1, delta_env_calls)
            policy_infer_time_avg = delta_infer_time / max(1, delta_infer_calls)
            update_time_avg = delta_update_time / max(1, delta_update_calls)
            valid_action_ratio = delta_mask_sum / max(1, delta_mask_calls)
            illegal_action_rate = delta_illegal / max(1, delta_env_calls)
            train_accounted = delta_env_step_time + delta_infer_time + delta_update_time
            time_accounting_coverage_train = train_accounted / max(1e-9, train_chunk_time)
            time_frac_env = delta_env_step_time / train_chunk_time if train_chunk_time > 0 else 0.0
            time_frac_infer = delta_infer_time / train_chunk_time if train_chunk_time > 0 else 0.0
            time_frac_update = delta_update_time / train_chunk_time if train_chunk_time > 0 else 0.0

            sys_metrics = {}
            if torch.cuda.is_available():
                alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
                sys_metrics["gpu_max_mem_allocated_mb"] = alloc
                sys_metrics["gpu_max_mem_reserved_mb"] = resv
                torch.cuda.reset_peak_memory_stats()

            circuit_stats_log = best_stats_candidate or best_stats_global or {}

            # metrics.jsonl/TB：本 chunk 统计
            metrics = {
                "trained_steps": trained,
                "eval": {  # 本次评估结果（行为 + 性能）
                    "success": succ,
                    "mean_fidelity": mean_F,
                    "best_fidelity_so_far": best_fidelity,
                    "best_qasm_available": best_qasm_global is not None,
                    "mask_valid_ratio": mask_ratio_mean,
                    "illegal_rate": illegal_rate_eval,
                    "mean_episode_time_sec": mean_ep_time_eval,
                    "mean_episode_steps": mean_ep_steps_eval,
                    "entropy_mean": entropy_mean_eval,
                    "p_max_mean": p_max_mean_eval,
                    "margin_mean": margin_mean_eval,
                    "k_eff_mean": k_eff_mean_eval,
                },
                "circuit": {  # 本次评估的电路结构
                    "gate_count": circuit_stats_log.get("gate_count"),
                    "depth": circuit_stats_log.get("depth"),
                    "cx_count": circuit_stats_log.get("cx_count"),
                    "oneq_count": circuit_stats_log.get("oneq_count"),
                    # 历史最短电路（按 gate_count 最小）
                    "min_gate_count_so_far": (min_stats_global or {}).get("gate_count"),
                    "min_depth_so_far": (min_stats_global or {}).get("depth"),
                },
                "time": {  # 本 chunk 训练/评估耗时
                    "train_chunk_sec": train_chunk_time,
                    "eval_sec": eval_time,
                },
                "perf": {  # 性能与时间拆解
                    "steps_per_sec": steps_per_sec,
                    "env_step_time_avg_sec": env_step_time_avg,
                    "policy_infer_time_avg_sec": policy_infer_time_avg,
                    "update_time_avg_sec": update_time_avg,
                    "time_accounting_coverage_train": time_accounting_coverage_train,
                    "time_frac_env": time_frac_env,
                    "time_frac_infer": time_frac_infer,
                    "time_frac_update": time_frac_update,
                },
                "mask": {  # 合法性/非法率
                    "valid_action_ratio": valid_action_ratio,
                    "illegal_action_rate": illegal_action_rate,
                },
            }
            if ppo_means:
                metrics["ppo"] = {
                    "approx_kl_mean": ppo_means.get("approx_kl"),
                    "clipfrac_mean": ppo_means.get("clipfrac"),
                    "entropy_mean": ppo_means.get("entropy"),
                    "loss_total_mean": ppo_means.get("loss_total"),
                    "loss_actor_mean": ppo_means.get("loss_actor"),
                    "loss_critic_mean": ppo_means.get("loss_critic"),
                }
            if sys_metrics:
                metrics["sys"] = sys_metrics

            # 索引文件：记录 trace / ckpt 路径
            try:
                chunk_entry = {
                    "chunk_idx": int(chunk_idx),
                    "trained_steps": int(trained),
                    "train_trace": os.path.relpath(train_trace_path, logger.run_dir),
                    "eval_trace": os.path.relpath(eval_trace_path, logger.run_dir),
                    "ckpt": os.path.relpath(ckpt_path, logger.run_dir),
                }
                with open(chunks_index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

            logger.log_metrics(trained, metrics)

            # ⚠️ 注意：正式实验时建议禁用早停（设置 early_stop_on_success=False）
            # 早停虽然节省训练预算，但可能在神经网络"刚学会完成任务"时就停止，
            # 错过后续"学会用更短电路完成任务"的阶段，从而削弱电路质量表现。
            if early_stop_on_success and consecutive_successes >= early_stop_consecutive:
                break

            chunk_idx += 1

    except Exception as e:
        print(f"[train] Error during training: {e}")
        raise
    finally:
        wall_time = time.time() - t0
        # 清理
        try:
            train_envs.close()
            eval_env.close()
        except Exception:
            pass

    summary = {
        "run_id": run_id,
        "scheme": scheme,                      # 新增：state representation 类型
        "encoder_type": encoder_type,          # 新增：encoder 类型
        "encoder_params": encoder_params,      # 新增：encoder 参数量
        "task_id": task.get("task_id", None),
        "difficulty_bin": task.get("difficulty_bin", None),
        "seed": int(seed),
        "budget_steps": int(total_budget_steps),
        "trained_steps": int(trained),
        "solved": int(solved),
        "steps_to_solve": (int(steps_to_solve) if steps_to_solve is not None else None),
        "best_fidelity": float(best_fidelity),   # 全程最佳（跨评估轮）
        "final_fidelity": float(final_fidelity), # 最后一次评估均值
        "wall_time_sec": float(wall_time),
        "time_to_solve_sec": float(t_solve) if t_solve is not None else wall_time * 10,
        "best_qasm": best_qasm_global,
        "solved_qasm": solved_qasm,
        "target_qasm": target_qasm,
        "best_gate_count": (best_stats_global or {}).get("gate_count"),
        "best_depth": (best_stats_global or {}).get("depth"),
        "best_cx_count": (best_stats_global or {}).get("cx_count"),
        "best_oneq_count": (best_stats_global or {}).get("oneq_count"),
        # 达到目标的最短电路统计（按 gate_count 最小）
        "min_gate_count": (min_stats_global or {}).get("gate_count"),
        "min_depth": (min_stats_global or {}).get("depth"),
        "min_cx_count": (min_stats_global or {}).get("cx_count"),
        "min_oneq_count": (min_stats_global or {}).get("oneq_count"),
        "min_qasm": min_qasm_global,
        "mean_episode_time_sec": float(mean_episode_time),
        "steps_per_sec": steps_per_sec if "steps_per_sec" in locals() else 0.0,  # 训练阶段
        "env_step_time_avg_sec": env_step_time_avg if "env_step_time_avg" in locals() else 0.0,
        "policy_infer_time_avg_sec": policy_infer_time_avg if "policy_infer_time_avg" in locals() else 0.0,
        "update_time_avg_sec": update_time_avg if "update_time_avg" in locals() else 0.0,
        "mask_valid_action_ratio": valid_action_ratio if "valid_action_ratio" in locals() else 0.0,
        "illegal_action_rate": illegal_action_rate if "illegal_action_rate" in locals() else 0.0,
        "final_entropy_mean": float(last_entropy_mean),
        "final_p_max_mean": float(last_p_max_mean),
        "final_margin_mean": float(last_margin_mean),
        "final_k_eff_mean": float(last_k_eff_mean),
    }
    logger.log_summary(summary)
    logger.close()

    return summary  # run 级最终结果（供上层汇总/排序）

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
            "mean_best_gate_count": None,
            "median_best_gate_count": None,
            "mean_best_depth": None,
            "median_best_depth": None,
            "mean_best_cx_count": None,
            "mean_best_oneq_count": None,
            "mean_episode_time_sec": None,
            "mean_time_per_step_sec": None,
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

    gate_counts = [r.get("best_gate_count") for r in records if r.get("best_gate_count") is not None]
    depth_counts = [r.get("best_depth") for r in records if r.get("best_depth") is not None]
    cx_counts = [r.get("best_cx_count") for r in records if r.get("best_cx_count") is not None]
    oneq_counts = [r.get("best_oneq_count") for r in records if r.get("best_oneq_count") is not None]

    mean_gate = float(np.mean(gate_counts)) if gate_counts else None
    median_gate = float(np.median(gate_counts)) if gate_counts else None
    mean_depth = float(np.mean(depth_counts)) if depth_counts else None
    median_depth = float(np.median(depth_counts)) if depth_counts else None
    mean_cx = float(np.mean(cx_counts)) if cx_counts else None
    mean_oneq = float(np.mean(oneq_counts)) if oneq_counts else None

    episode_times = [r.get("mean_episode_time_sec") for r in records if r.get("mean_episode_time_sec") is not None]
    mean_episode_time = float(np.mean(episode_times)) if episode_times else None

    time_per_step = []
    for r in records:
        budget = float(r.get("budget_steps", 0.0))
        wt = float(r.get("wall_time_sec", 0.0))
        if budget > 0 and np.isfinite(wt):
            time_per_step.append(wt / budget)
    mean_time_per_step = float(np.mean(time_per_step)) if time_per_step else None
    
    return {
        "n_runs": int(len(records)),
        "success_rate": success_rate,
        "mean_steps_to_solve": mean_steps,
        "median_steps_to_solve": median_steps,
        "mean_best_fidelity": best_F,
        "mean_final_fidelity": final_F,
        "mean_wall_time_sec": wall_t,
        "mean_time_to_solve_sec": time_to_solve_t,
        "mean_best_gate_count": mean_gate,
        "median_best_gate_count": median_gate,
        "mean_best_depth": mean_depth,
        "median_best_depth": median_depth,
        "mean_best_cx_count": mean_cx,
        "mean_best_oneq_count": mean_oneq,
        "mean_episode_time_sec": mean_episode_time,
        "mean_time_per_step_sec": mean_time_per_step,
    }


def compute_weighted_scores(summaries: list, weights=None):
    """
    在 suite 级对候选进行带权归一化打分（BenchRL-QAS 风格）。
    E=1-bestF，G=gate，D=depth，T=episode_time（或 step time）。
    """
    if weights is None:
        weights = {"E": 0.5, "G": 0.2, "D": 0.2, "T": 0.1}

    metric_values = {
        "E": [1 - s.get("mean_best_fidelity", 0.0) for s in summaries],
        "G": [s.get("mean_best_gate_count") for s in summaries],
        "D": [s.get("mean_best_depth") for s in summaries],
        "T": [s.get("mean_episode_time_sec") or s.get("mean_time_per_step_sec") for s in summaries],
    }

    normed = {}
    for key, vals in metric_values.items():
        finite_vals = [v for v in vals if v is not None]
        if not finite_vals:
            normed[key] = [None for _ in vals]
            continue
        vmin, vmax = min(finite_vals), max(finite_vals)
        if abs(vmax - vmin) < 1e-9:
            normed[key] = [0.0 if v is not None else None for v in vals]
        else:
            normed[key] = [((v - vmin) / (vmax - vmin)) if v is not None else None for v in vals]

    for i, s in enumerate(summaries):
        e_norm = normed["E"][i]
        g_norm = normed["G"][i]
        d_norm = normed["D"][i]
        t_norm = normed["T"][i]
        weighted = 0.0
        for k, w in weights.items():
            val = {"E": e_norm, "G": g_norm, "D": d_norm, "T": t_norm}.get(k, None)
            if val is not None:
                weighted += w * val
        s["weighted_score"] = weighted
        s["E_norm"] = e_norm
        s["G_norm"] = g_norm
        s["D_norm"] = d_norm
        s["T_norm"] = t_norm

    return summaries


def write_suite_summary_csv(stage_name: str, scheme: str, summaries: list, out_dir: str = "logs", filename: str = "suite_summary.csv"):
    if not summaries:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    headers = [
        "stage",
        "scheme",
        "net_cfg_name",
        "success_rate",
        "mean_best_fidelity",
        "mean_final_fidelity",
        "mean_steps_to_solve",
        "median_steps_to_solve",
        "mean_best_gate_count",
        "median_best_gate_count",
        "mean_best_depth",
        "median_best_depth",
        "mean_episode_time_sec",
        "mean_time_per_step_sec",
        "mean_wall_time_sec",
        "weighted_score",
        "E_norm",
        "G_norm",
        "D_norm",
        "T_norm",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for s in summaries:
            row = [
                stage_name,
                scheme,
                s.get("net_cfg_name"),
                s.get("success_rate"),
                s.get("mean_best_fidelity"),
                s.get("mean_final_fidelity"),
                s.get("mean_steps_to_solve"),
                s.get("median_steps_to_solve"),
                s.get("mean_best_gate_count"),
                s.get("median_best_gate_count"),
                s.get("mean_best_depth"),
                s.get("median_best_depth"),
                s.get("mean_episode_time_sec"),
                s.get("mean_time_per_step_sec"),
                s.get("mean_wall_time_sec"),
                s.get("weighted_score"),
                s.get("E_norm"),
                s.get("G_norm"),
                s.get("D_norm"),
                s.get("T_norm"),
            ]
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    return path

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
                stage_name=stage_name,
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

    # 4) ranking （带权 + 兼容旧策略）
    all_summaries = compute_weighted_scores(all_summaries)

    def rank_key(s):
        steps = s["mean_steps_to_solve"]
        steps_key = steps if steps is not None else 1e18
        ws = s.get("weighted_score", None)
        ws_key = ws if ws is not None else 1e9
        return (ws_key, -s["success_rate"], steps_key, -s["mean_best_fidelity"], s["mean_wall_time_sec"])

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
                f"gate={s.get('mean_best_gate_count')}  "
                f"depth={s.get('mean_best_depth')}  "
                f"T={s.get('mean_episode_time_sec')}  "
                f"Wscore={s.get('weighted_score'):.4f}  "
                f"time={s['mean_wall_time_sec']:.1f}s"
            )

    # 5) 写 suite summary CSV
    log_dir = train_cfg.get("log_dir", "logs")
    csv_path = write_suite_summary_csv(
        stage_name=suite_name,
        scheme=scheme,
        summaries=all_summaries_sorted,
        out_dir=os.path.join(log_dir, str(suite_name)),
            )

    return {
        "sample_summary": sample_summary,
        "sampled_tasks": sampled_tasks,
        "summaries_sorted": all_summaries_sorted,
        "summaries_raw": all_summaries,
        "records_by_cfg": all_records_by_cfg,
        "suite_summary_csv": csv_path,
    }




def load_task_by_id(in_dir: str, suite_name: str, task_id: str) -> Dict[str, Any]:
    """从指定任务集加载并返回单个 task（按 task_id 匹配）。"""
    tasks = normalize_tasks(load_task_suite(in_dir, suite_name))
    matches = [t for t in tasks if str(t.get("task_id")) == str(task_id)]
    if not matches:
        raise KeyError(f"task_id={task_id} not found in suite {suite_name}")
    if len(matches) > 1:
        print(f"[warn] found {len(matches)} tasks with task_id={task_id}, use the first one.")
    return matches[0]


def load_sampled_tasks_by_bin(
    *,
    in_dir: str,
    suite_name: str,
    sampled_json_path: str,
) -> Dict[str, list]:
    """
    读取按 difficulty_bin 抽样得到的 ID 列表（如 Final_sampled_bin10.json），
    返回一个大字典：bin -> [task dict...]
    random 与 common 的任务都会被映射回完整 task。
    """
    tasks = normalize_tasks(load_task_suite(in_dir, suite_name))
    id_map = {str(t.get("task_id")): t for t in tasks}

    data = json.load(open(sampled_json_path, "r", encoding="utf-8"))
    sampled_ids = data.get("sampled", {})

    out: Dict[str, list] = {}
    for bin_name, ids in sampled_ids.items():
        out[bin_name] = []
        for tid in ids:
            tid_str = str(tid)
            if tid_str not in id_map:
                raise KeyError(f"task_id={tid_str} not found in suite {suite_name}")
            out[bin_name].append(id_map[tid_str])

    return out


def run_task_by_id(
    *,
    in_dir: str,
    suite_name: str,
    task_id: str,
    scheme: str,
    net_cfg_list: list,
    algo_cfg: dict,
    train_cfg: dict,
    seeds: list,
    device: str,
    verbose: bool = True,
):
    """
    入口：指定任务集 + task_id，只运行这个任务的训练与评估。
    用法示例：
      run_task_by_id(
          in_dir="task_suites",
          suite_name="Dev",
          task_id="task_22",
          scheme="gate_seq",
          net_cfg_list=[...],
          algo_cfg=algo_cfg,
          train_cfg=train_cfg,
          seeds=[0],
          device="cuda",
      )
    """
    task = load_task_by_id(in_dir, suite_name, task_id)
    all_summaries = []
    all_records_by_cfg = {}
    for net_cfg in net_cfg_list:
        records, summary = run_cfg_on_taskset(
            stage_name=suite_name,
            tasks=[task],
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
    return {"summaries": all_summaries, "records_by_cfg": all_records_by_cfg}

def run_suite_all_tasks(
    *,

    in_dir: str,
    suite_name: str,
    scheme: str,
    net_cfg_list: list,
    algo_cfg: dict,
    train_cfg: dict,
    seeds: list,
    device: str,
    verbose: bool = True,
    tasks_set = None,
):
    """
    在指定任务集上运行所有任务（不抽样），按 net_cfg 列出总结并排序。
    复用 run_cfg_on_taskset，排序规则与 run_stage 保持一致。
    """
    if tasks_set is None:
        tasks = normalize_tasks(load_task_suite(in_dir, suite_name))
    else:
        tasks = tasks_set

    if verbose:
        print(f"\n=== Run full suite {suite_name} / scheme={scheme} ===")
        print("All task_ids:", [t.get("task_id") for t in tasks])

    all_summaries = []
    all_records_by_cfg = {}

    for net_cfg in net_cfg_list:
        records, summary = run_cfg_on_taskset(
            stage_name=suite_name,
            tasks=tasks,
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

    all_summaries = compute_weighted_scores(all_summaries)

    def rank_key(s):
        steps = s["mean_steps_to_solve"]
        steps_key = steps if steps is not None else 1e18
        ws = s.get("weighted_score", None)
        ws_key = ws if ws is not None else 1e9
        return (ws_key, -s["success_rate"], steps_key, -s["mean_best_fidelity"], s["mean_wall_time_sec"])

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
                f"gate={s.get('mean_best_gate_count')}  "
                f"depth={s.get('mean_best_depth')}  "
                f"T={s.get('mean_episode_time_sec')}  "
                f"Wscore={s.get('weighted_score'):.4f}  "
                f"time={s['mean_wall_time_sec']:.1f}s"
            )

    csv_path = write_suite_summary_csv(
        stage_name=suite_name,
        scheme=scheme,
        summaries=all_summaries_sorted,
        out_dir=os.path.join(train_cfg.get("log_dir", "logs"), str(suite_name)),
            )

    return {
        "tasks": tasks,
        "summaries_sorted": all_summaries_sorted,
        "summaries_raw": all_summaries,
        "records_by_cfg": all_records_by_cfg,
        "suite_summary_csv": csv_path,
    }



def main(
scheme: str, 
net_cfg_list: list, 
algo_cfg: dict, 
train_cfg: dict, 
seeds: list, 
in_dir: str = "task_suites", 
suite_name: str = "Final", 
sampled_json_path: str = "task_suites/Final_sampled_bin10.json",
):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheme = scheme
    
    # ----------------
    # 1) network configs (baseline grid)
    #   说明：net_cfg 通过 build_encoder + SharedMLP 共同生效
    #   注意：net_cfg["encoder"] 必须与 scheme 兼容（见 SCHEME_ENCODER_COMPAT）
    # ----------------
    if net_cfg_list is None:
        # 根据 scheme 选择默认的 encoder 配置
        default_encoder = SCHEME_ENCODER_COMPAT.get(scheme, ["mlp"])[0]
        net_cfg_list_final = [
            {"name": f"{default_encoder.upper()}_default",
             "encoder": default_encoder,
             "hid": 64, "depth": 2,
             "act": "silu", "use_ln": True,
             "dropout": 0.0, "input_dropout": 0.0,
             "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},
        ]
    else:
        net_cfg_list_final = net_cfg_list
    # ----------------
    # 2) PPO algo cfg (两个版本：标准 / 更强探索)
    # ----------------
    if algo_cfg is None:
        algo_cfg_final = {
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
    else:
        algo_cfg_final = algo_cfg
    

    # ----------------
    # 3) train cfg (Dev：先跑通 + 快速出信号)
    #   注意：run_one_task 里每 eval_every_steps 会重新建一次 on-policy buffer
    # ----------------
    if train_cfg is None:
        train_cfg_final = {
            "n_train_env": 1,  # 使用单环境避免 trace 数据对齐问题
            "n_test_env": 8, 
            "n_eval_env": 8, 
            "n_eval_episodes": 100, 
            "n_eval_steps": 100, 
        }
    else:
        train_cfg_final = train_cfg

    # ----------------
    # 4) seeds（先 smoke：1 个；跑 baseline：2~3 个）
    # ----------------
    if seeds is None:
        seeds_final = [0, 1, 2, 3, 4]
    else:
        seeds_final = seeds

    # ----------------
    # 5) 日志：stdout/stderr 同时写到文件，便于后续查看
    # ----------------
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 定制日志文件名（同目录下），便于直接查阅；加时间戳避免覆盖
    log_path = os.path.join(log_dir, f"{ts}_log.log")

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
            return len(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w", encoding="utf-8") as f, \
         redirect_stdout(Tee(sys.stdout, f)), \
         redirect_stderr(Tee(sys.stderr, f)):
        
        print(f"[log] writing to {log_path}")
        ## task suite preparation
        tasks = load_sampled_tasks_by_bin(
            in_dir=in_dir,
            suite_name=suite_name,
            sampled_json_path=sampled_json_path,)

        for bin_name, task_list in tasks.items():

    
            run_suite_all_tasks(
                in_dir=in_dir,
                suite_name=suite_name,
                scheme=scheme,
                net_cfg_list=net_cfg_list_final,
                algo_cfg=algo_cfg_final,
                train_cfg=train_cfg_final,
                seeds=seeds_final,
                device=device,
                verbose=True,
                tasks_set=task_list,
            )
    

def main_multi_schemes(
    experiments: List[Tuple[str, List[dict]]],
    algo_cfg: dict,
    train_cfg: dict,
    seeds: List[int],
    in_dir: str = "task_suites",
    suite_name: str = "Final",
    sampled_json_path: str = "task_suites/Final_sampled_bin10.json",
):
    """
    批量运行多种 scheme + encoder 组合的实验。
    
    Args:
        experiments: List of (scheme, net_cfg_list) tuples
            例如:
            [
                ("gate_seq", [{"name": "MLP_100k", "encoder": "mlp", ...}]),
                ("gate_seq", [{"name": "RNN_100k", "encoder": "rnn", ...}]),
                ("2d_grid",  [{"name": "CNN_100k", "encoder": "cnn", ...}]),
                ("graph",    [{"name": "GIN_100k", "encoder": "gin", ...}]),
            ]
        algo_cfg: PPO 算法配置
        train_cfg: 训练配置
        seeds: 随机种子列表
        in_dir: 任务集目录
        suite_name: 任务集名称
        sampled_json_path: 抽样任务 JSON 路径
    
    Examples:
        >>> experiments = [
        ...     ("gate_seq", [{"name": "MLP_100k", "encoder": "mlp", "hid": 48, ...}]),
        ...     ("gate_seq", [{"name": "RNN_100k", "encoder": "rnn", "hidden_size": 126, ...}]),
        ...     ("2d_grid",  [{"name": "CNN_100k", "encoder": "cnn", "hid": 94, ...}]),
        ...     ("graph",    [{"name": "GIN_100k", "encoder": "gin", "hid": 110, ...}]),
        ... ]
        >>> main_multi_schemes(experiments, algo_cfg, train_cfg, seeds=[0, 1, 2])
    """
    print(f"[multi_schemes] Total experiments: {len(experiments)}")
    
    for exp_idx, (scheme, net_cfg_list) in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"[Experiment {exp_idx + 1}/{len(experiments)}] scheme={scheme}")
        print(f"  encoders: {[cfg.get('name', cfg.get('encoder', 'unnamed')) for cfg in net_cfg_list]}")
        print("=" * 60)
        
        main(
            scheme=scheme,
            net_cfg_list=net_cfg_list,
            algo_cfg=algo_cfg,
            train_cfg=train_cfg,
            seeds=seeds,
            in_dir=in_dir,
            suite_name=suite_name,
            sampled_json_path=sampled_json_path,
        )
    
    print(f"\n[multi_schemes] All {len(experiments)} experiments completed.")


if __name__ == "__main__":
    # main(
    # scheme = "gate_seq", 
    # net_cfg_list = [
    # # ====== Tiny / Smoke ======
    # {"name": "E055k_h64_d2",
    #  "hid": 64, "depth": 2,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== Small ======
    # {"name": "E118k_h128_d2",
    #  "hid": 128, "depth": 2,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== Medium ======
    # {"name": "E268k_h256_d2",
    #  "hid": 256, "depth": 2,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== Depth control (only one) ======
    # {"name": "E334k_h256_d3",
    #  "hid": 256, "depth": 3,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== Large ======
    # {"name": "E451k_h384_d2",
    #  "hid": 384, "depth": 2,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== XXL ======
    # {"name": "E599k_h384_d3",
    #  "hid": 384, "depth": 3,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},
    
    # # ====== XL ======
    # {"name": "E667k_h512_d2",
    #  "hid": 512, "depth": 2,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},

    # # ====== XXL ======
    # {"name": "E930k_h512_d3",
    #  "hid": 512, "depth": 3,
    #  "act": "silu", "use_ln": True,
    #  "dropout": 0.0, "input_dropout": 0.0,
    #  "out_dim": 256, "shared_out_dim": 256, "shared_act": "silu"},
    #  ],
    # algo_cfg = {
    #     "lr": 3e-4,
    #     "gamma": 0.99,
    #     "gae_lambda": 0.95,
    #     "max_grad_norm": 0.5,
    #     "eps_clip": 0.2,
    #     "vf_coef": 0.5,
    #     "ent_coef": 0.02,  # 比你原来的 0.01 更鼓励探索（Dev 阶段更有用）
    #     "advantage_normalization": True,
    #     "value_clip": False,
    #     "return_scaling": False,
    # }, 
    # train_cfg= {
    #     "n_train_env": 8, 
    #     # 先别太大：调 bug / 看趋势用 30k~50k 更合适
    #     "total_budget_steps": 500000, 
    #     # 每 5k 做一次 greedy_eval（steps_to_solve 粒度也就是 5k）
    #     "eval_every_steps": 5000,
    #     "eval_episodes": 5, ## 不应该是1，有些参数每个espisode都不一样
    #     # PPO rollout / update 强度：先保守，稳定优先
    #     "collect_steps": 2048,
    #     "update_reps": 8,
    #     "batch_size": 256,
    #     # buffer_size 只是下限，代码里会用 max(buffer_size, this_chunk*2) 兜底
    #     "buffer_size": 20000,
    #     # 需要的话你也能显式写死阈值（否则从 env 里读）
    #     "fidelity_threshold": 0.95,
    #     # 可选：覆盖 env 默认 max_gates（不写则用 env 默认 78）
    #     "max_gates": 25,  ##最大14 1.5 * 14 = 20
    #     # 评估成功后直接提前停掉，节省预算
    #     "early_stop_on_success": True,
    #     # 连续成功多少次才提前停（>=1）
    #     "early_stop_consecutive_success": 3,
    # }, 
    #     seeds=[0, 1, 2, 3, 4], 
    #     in_dir="task_suites", 
    #     suite_name="Final", 
    #     sampled_json_path="task_suites/Final_sampled_bin10.json"
    #     )


    # ============================================================================
    # 全面的 State Scheme Benchmark 实验
    # 涵盖所有 scheme + encoder 组合，每种 2 个配置（小/中规模）
    # ============================================================================
    
    # 共享的算法配置
    algo_cfg = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "eps_clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.02,
        "advantage_normalization": True,
        "value_clip": False,
        "return_scaling": False,
    }
    
    # 共享的训练配置
    train_cfg = {
        # ===== 日志目录 =====
        "log_dir": "logs_benchmark_schemes",
        "n_train_env": 1,  # 使用单环境避免 trace 数据对齐问题
        # 先别太大：调 bug / 看趋势用 30k~50k 更合适
        "total_budget_steps": 10240, 
        # 每 5k 做一次 greedy_eval（steps_to_solve 粒度也就是 5k）
        "eval_every_steps": 512,
        "eval_episodes": 2,  # 评估 episode 数
        # PPO rollout / update 强度：先保守，稳定优先
        "collect_steps": 2048,
        "update_reps": 8,
        "batch_size": 256,
        # buffer_size 只是下限，代码里会用 max(buffer_size, this_chunk*2) 兜底
        "buffer_size": 1024,
        # 需要的话你也能显式写死阈值（否则从 env 里读）
        "fidelity_threshold": 0.95,
        # 可选：覆盖 env 默认 max_gates（不写则用 env 默认 78）
        "max_gates": 25,  # 最大14 * 1.4 = 20
        # 评估成功后直接提前停掉，节省预算
        "early_stop_on_success": True,
        # 连续成功多少次才提前停（>=1）
        "early_stop_consecutive_success": 3,
    }
    
    # 共享的 SharedMLP 配置
    shared_cfg = {"shared_out_dim": 256, "shared_act": "silu"}
    
    # ============================================================================
    # 实验配置：(scheme, net_cfg_list) 列表
    # ============================================================================
    experiments = [
        # ===== 1. gate_seq + MLP =====
        ("gate_seq", [
            {"name": "MLP_small_h64_d2", "encoder": "mlp",
             "hid": 64, "depth": 2, "out_dim": 128,
             "act": "silu", "use_ln": True, "dropout": 0.0,
             **shared_cfg},
            {"name": "MLP_medium_h128_d2", "encoder": "mlp",
             "hid": 128, "depth": 2, "out_dim": 256,
             "act": "silu", "use_ln": True, "dropout": 0.0,
             **shared_cfg},
        ]),
        
        # ===== 2. gate_seq + RNN =====
        ("gate_seq", [
            {"name": "RNN_small_hs64", "encoder": "rnn",
             "embed_dim": 32, "hidden_size": 64, "num_layers": 1,
             "rnn_type": "lstm", "bidirectional": False, "pool": "last",
             "out_dim": 64,
             **shared_cfg},
            {"name": "RNN_medium_hs128", "encoder": "rnn",
             "embed_dim": 64, "hidden_size": 128, "num_layers": 1,
             "rnn_type": "lstm", "bidirectional": False, "pool": "last",
             "out_dim": 128,
             **shared_cfg},
        ]),
        
        # ===== 3. 2d_grid + CNN =====
        ("2d_grid", [
            {"name": "CNN_2d_small_h48_d3", "encoder": "cnn",
             "hid": 48, "depth": 3, "kernel_size": 3,
             "out_dim": 96, "use_proj": True,
             "act": "relu", "dropout": 0.0, "mode": "grid",
             **shared_cfg},
            {"name": "CNN_2d_medium_h96_d3", "encoder": "cnn",
             "hid": 96, "depth": 3, "kernel_size": 3,
             "out_dim": 192, "use_proj": True,
             "act": "relu", "dropout": 0.0, "mode": "grid",
             **shared_cfg},
        ]),
        
        # ===== 4. 3d_tensor + CNN =====
        ("3d_tensor", [
            {"name": "CNN_3d_small_h48_d3", "encoder": "cnn",
             "hid": 48, "depth": 3, "kernel_size": 3,
             "out_dim": 96, "use_proj": True,
             "act": "relu", "dropout": 0.0, "mode": "tensor",
             **shared_cfg},
            {"name": "CNN_3d_medium_h96_d3", "encoder": "cnn",
             "hid": 96, "depth": 3, "kernel_size": 3,
             "out_dim": 192, "use_proj": True,
             "act": "relu", "dropout": 0.0, "mode": "tensor",
             **shared_cfg},
        ]),
        
        # ===== 5. graph + GIN =====
        ("graph", [
            {"name": "GIN_small_h64_d3", "encoder": "gin",
             "hid": 64, "depth": 3, "mlp_depth": 2,
             "out_dim": 128, "readout": "mean",
             "norm": "ln", "dropout": 0.0,
             **shared_cfg},
            {"name": "GIN_medium_h128_d3", "encoder": "gin",
             "hid": 128, "depth": 3, "mlp_depth": 2,
             "out_dim": 256, "readout": "mean",
             "norm": "ln", "dropout": 0.0,
             **shared_cfg},
        ]),
    ]
    
    # 运行批量实验 (测试用小规模配置)
    # print("Running test experiments...")
    # main_multi_schemes(
    #     experiments=experiments,
    #     algo_cfg=algo_cfg,
    #     train_cfg=train_cfg,
    #     seeds=[0, 1],
    #     in_dir="task_suites",
    #     suite_name="Final",
    #     sampled_json_path="task_suites/Final_sampled_bin5.json",
    # )
    # print(" test experiments finished...")

    # ============================================================================
    # 正式实验：gate_seq + MLP（8 个参数量等级）
    # ============================================================================
    
    # PPO 算法配置
    mlp_algo_cfg = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "eps_clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.02,
        "advantage_normalization": True,
        "value_clip": False,
        "return_scaling": False,
    }
    
    # 训练配置
    mlp_train_cfg = {
        "log_dir": "logs_benchmark_MLP",
        "n_train_env": 1,
        "total_budget_steps": 200000,   # 200k步，配合early_stop足够
        "eval_every_steps": 2000,       # 每2k步评估，共100次评估点
        "eval_episodes": 5,
        "collect_steps": 2048,
        "update_reps": 8,
        "batch_size": 256,
        "buffer_size": 10000,           # 缩小buffer，200k步够用
        "fidelity_threshold": 0.95,
        "max_gates": 25,  # 当前任务最大门数：14，最大预算steps：1.5 * 14 = 21，向上取整25
        "early_stop_on_success": True,
        "early_stop_consecutive_success": 3,
    }
    
    # SharedMLP 配置
    mlp_shared_cfg = {"shared_out_dim": 256, "shared_act": "silu"}
    
    # MLP 实验配置（4 个代表性参数量等级）
    all_mlp_cfgs = [
        # 0: ~186K params (hid=64, depth=2) - 小型
        {"name": "MLP_h64_d2", "encoder": "mlp",
         "hid": 64, "depth": 2, "out_dim": 256,
         "act": "silu", "use_ln": True, "dropout": 0.0,
         **mlp_shared_cfg},
        # 1: ~380K params (hid=128, depth=2) - 中小型
        {"name": "MLP_h128_d2", "encoder": "mlp",
         "hid": 128, "depth": 2, "out_dim": 256,
         "act": "silu", "use_ln": True, "dropout": 0.0,
         **mlp_shared_cfg},
        # 2: ~858K params (hid=256, depth=3) - 中型
        {"name": "MLP_h256_d3", "encoder": "mlp",
         "hid": 256, "depth": 3, "out_dim": 256,
         "act": "silu", "use_ln": True, "dropout": 0.0,
         **mlp_shared_cfg},
        # 3: ~1978K params (hid=512, depth=3) - 大型
        {"name": "MLP_h512_d3", "encoder": "mlp",
         "hid": 512, "depth": 3, "out_dim": 256,
         "act": "silu", "use_ln": True, "dropout": 0.0,
         **mlp_shared_cfg},
    ]
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="MLP Benchmark Training")
    parser.add_argument("--arch", type=int, default=None,
                        help="架构索引 0-3: 0=h64_d2, 1=h128_d2, 2=h256_d3, 3=h512_d3")
    args = parser.parse_args()
    
    # 选择要运行的架构
    if args.arch is not None:
        if not 0 <= args.arch < len(all_mlp_cfgs):
            raise ValueError(f"--arch 必须在 0-{len(all_mlp_cfgs)-1} 之间")
        selected = [all_mlp_cfgs[args.arch]]
        print(f"[单架构模式] 运行: {selected[0]['name']}")
    else:
        selected = all_mlp_cfgs
        print(f"[顺序模式] 依次运行全部 {len(all_mlp_cfgs)} 个架构")
    
    mlp_experiments = [("gate_seq", selected)]
    
    # 运行 MLP 实验
    main_multi_schemes(
        experiments=mlp_experiments,
        algo_cfg=mlp_algo_cfg,
        train_cfg=mlp_train_cfg,
        seeds=[0, 1, 2, 3, 4],
        in_dir="task_suites",
        suite_name="Final",
        sampled_json_path="task_suites/final_tasks.json",
    )

    # ============================================================================
    # 其他 Schemes 实验配置（参数量对齐 MLP 的 4 个等级）
    # 目标参数量: ~186K, ~380K, ~858K, ~1978K (±10%)
    # ============================================================================
    
    # 共享配置（与 MLP 相同）
    shared_cfg = {"shared_out_dim": 256, "shared_act": "silu"}
    
    # --------------------------------------------------------------------------
    # 1. RNN 配置 (gate_seq scheme)
    # 注意：
    #   - post_ln/post_act/post_dropout 由 SharedMLP 负责，encoder 不处理
    #   - out_dim = hidden_size，禁用 encoder 内部 projection，由 SharedMLP 投影
    #   - 参数量 = Embedding + LSTM (不含 projection)
    # --------------------------------------------------------------------------
    all_rnn_cfgs = [
        # 0: ~186K params - 小型
        # Embedding: 33*64=2.1K, LSTM: 4*184*(64+184)+8*184=184K
        {"name": "RNN_h184_e64", "encoder": "rnn",
         "embed_dim": 64, "hidden_size": 184, "num_layers": 1,
         "rnn_type": "lstm", "bidirectional": False, "pool": "last",
         "out_dim": 184,  # = hidden_size, 禁用 proj
         "post_ln": False, "post_act": "none", "post_dropout": 0.0,
         **shared_cfg},
        # 1: ~380K params - 中小型
        # Embedding: 33*96=3.2K, LSTM: 4*262*(96+262)+8*262=377K
        {"name": "RNN_h262_e96", "encoder": "rnn",
         "embed_dim": 96, "hidden_size": 262, "num_layers": 1,
         "rnn_type": "lstm", "bidirectional": False, "pool": "last",
         "out_dim": 262,  # = hidden_size, 禁用 proj
         "post_ln": False, "post_act": "none", "post_dropout": 0.0,
         **shared_cfg},
        # 2: ~856K params - 中型
        # Embedding: 33*128=4.2K, LSTM: 4*401*(128+401)+8*401=852K
        {"name": "RNN_h401_e128", "encoder": "rnn",
         "embed_dim": 128, "hidden_size": 401, "num_layers": 1,
         "rnn_type": "lstm", "bidirectional": False, "pool": "last",
         "out_dim": 401,  # = hidden_size, 禁用 proj
         "post_ln": False, "post_act": "none", "post_dropout": 0.0,
         **shared_cfg},
        # 3: ~1.98M params - 大型
        # Embedding: 33*160=5.3K, LSTM: 4*626*(160+626)+8*626=1973K
        {"name": "RNN_h626_e160", "encoder": "rnn",
         "embed_dim": 160, "hidden_size": 626, "num_layers": 1,
         "rnn_type": "lstm", "bidirectional": False, "pool": "last",
         "out_dim": 626,  # = hidden_size, 禁用 proj
         "post_ln": False, "post_act": "none", "post_dropout": 0.0,
         **shared_cfg},
    ]
    
    # --------------------------------------------------------------------------
    # 2. CNN 2d_grid 配置 (1D CNN for 2d_grid scheme)
    # 注意：use_proj=False，禁用内部 projection，由 SharedMLP 投影
    # 参数量 = first_conv + depth * backbone_conv (不含 projection)
    # --------------------------------------------------------------------------
    all_cnn_2d_cfgs = [
        # 0: ~186K params - 小型
        # first: 4*143*3+143=1.9K, backbone: 3*(143²*3+143)=184K
        {"name": "CNN_2d_h143_d3", "encoder": "cnn",
         "hid": 143, "depth": 3, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 143
         "act": "silu", "dropout": 0.0, "mode": "grid",
         **shared_cfg},
        # 1: ~382K params - 中小型
        # first: 4*205*3+205=2.7K, backbone: 3*(205²*3+205)=379K
        {"name": "CNN_2d_h205_d3", "encoder": "cnn",
         "hid": 205, "depth": 3, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 205
         "act": "silu", "dropout": 0.0, "mode": "grid",
         **shared_cfg},
        # 2: ~860K params - 中型
        # first: 4*267*3+267=3.5K, backbone: 4*(267²*3+267)=857K
        {"name": "CNN_2d_h267_d4", "encoder": "cnn",
         "hid": 267, "depth": 4, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 267
         "act": "silu", "dropout": 0.0, "mode": "grid",
         **shared_cfg},
        # 3: ~1.98M params - 大型
        # first: 4*363*3+363=4.7K, backbone: 5*(363²*3+363)=1978K
        {"name": "CNN_2d_h363_d5", "encoder": "cnn",
         "hid": 363, "depth": 5, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 363
         "act": "silu", "dropout": 0.0, "mode": "grid",
         **shared_cfg},
    ]
    
    # --------------------------------------------------------------------------
    # 3. CNN 3d_tensor 配置 (2D CNN for 3d_tensor scheme)
    # 注意：use_proj=False，禁用内部 projection，由 SharedMLP 投影
    # 参数量 = first_conv2d + depth * backbone_conv2d (不含 projection)
    # 假设 in_channels=7 (typical), kernel=3x3
    # --------------------------------------------------------------------------
    all_cnn_3d_cfgs = [
        # 0: ~187K params - 小型
        # first: 7*82*9+82=5.2K, backbone: 3*(82²*9+82)=182K
        {"name": "CNN_3d_h82_d3", "encoder": "cnn",
         "hid": 82, "depth": 3, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 82
         "act": "silu", "dropout": 0.0, "mode": "tensor",
         **shared_cfg},
        # 1: ~384K params - 中小型
        # first: 7*118*9+118=7.6K, backbone: 3*(118²*9+118)=376K
        {"name": "CNN_3d_h118_d3", "encoder": "cnn",
         "hid": 118, "depth": 3, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 118
         "act": "silu", "dropout": 0.0, "mode": "tensor",
         **shared_cfg},
        # 2: ~864K params - 中型
        # first: 7*154*9+154=9.9K, backbone: 4*(154²*9+154)=854K
        {"name": "CNN_3d_h154_d4", "encoder": "cnn",
         "hid": 154, "depth": 4, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 154
         "act": "silu", "dropout": 0.0, "mode": "tensor",
         **shared_cfg},
        # 3: ~1.98M params - 大型
        # first: 7*209*9+209=13.4K, backbone: 5*(209²*9+209)=1967K
        {"name": "CNN_3d_h209_d5", "encoder": "cnn",
         "hid": 209, "depth": 5, "kernel_size": 3,
         "use_proj": False,  # out_dim = hid = 209
         "act": "silu", "dropout": 0.0, "mode": "tensor",
         **shared_cfg},
    ]
    
    # --------------------------------------------------------------------------
    # 4. GIN 配置 (graph scheme)
    # 注意：use_post_block=False，禁用 out_proj 和 post，由 SharedMLP 处理
    # 参数量 = in_proj (LazyLinear) + depth * GIN_layer (不含 out_proj/post)
    # 假设 node_feat_dim=24 (typical)
    # --------------------------------------------------------------------------
    all_gin_cfgs = [
        # 0: ~188K params - 小型
        # in_proj: 25*174=4.4K, GIN: 3*(2*174²+4*174+1)=184K
        {"name": "GIN_h174_d3", "encoder": "gin",
         "hid": 174, "depth": 3, "mlp_depth": 2,
         "readout": "mean", "norm": "ln", "act": "silu", "dropout": 0.0,
         "use_post_block": False,  # out_dim = hid = 174
         **shared_cfg},
        # 1: ~381K params - 中小型
        # in_proj: 25*249=6.2K, GIN: 3*(2*249²+4*249+1)=375K
        {"name": "GIN_h249_d3", "encoder": "gin",
         "hid": 249, "depth": 3, "mlp_depth": 2,
         "readout": "mean", "norm": "ln", "act": "silu", "dropout": 0.0,
         "use_post_block": False,  # out_dim = hid = 249
         **shared_cfg},
        # 2: ~862K params - 中型
        # in_proj: 25*376=9.4K, GIN: 3*(2*376²+4*376+1)=853K
        {"name": "GIN_h376_d3", "encoder": "gin",
         "hid": 376, "depth": 3, "mlp_depth": 2,
         "readout": "mean", "norm": "ln", "act": "silu", "dropout": 0.0,
         "use_post_block": False,  # out_dim = hid = 376
         **shared_cfg},
        # 3: ~1.98M params - 大型
        # in_proj: 25*495=12.4K, GIN: 4*(2*495²+4*495+1)=1968K
        {"name": "GIN_h495_d4", "encoder": "gin",
         "hid": 495, "depth": 4, "mlp_depth": 2,
         "readout": "mean", "norm": "ln", "act": "silu", "dropout": 0.0,
         "use_post_block": False,  # out_dim = hid = 495
         **shared_cfg},
    ]
    
    # --------------------------------------------------------------------------
    # 组合所有 schemes 的实验配置
    # --------------------------------------------------------------------------
    all_scheme_experiments = [
        ("gate_seq", all_rnn_cfgs),      # RNN
        ("2d_grid", all_cnn_2d_cfgs),    # CNN 1D
        ("3d_tensor", all_cnn_3d_cfgs),  # CNN 2D
        ("graph", all_gin_cfgs),         # GIN
    ]
    
    # --------------------------------------------------------------------------
    # 运行其他 schemes 实验（取消注释以运行）
    # --------------------------------------------------------------------------
    # for scheme, cfgs in all_scheme_experiments:
    #     print(f"\n{'='*60}")
    #     print(f"Running {scheme} experiments...")
    #     print(f"{'='*60}")
    #     main_multi_schemes(
    #         experiments=[(scheme, cfgs)],
    #         algo_cfg=mlp_algo_cfg,
    #         train_cfg={**mlp_train_cfg, "log_dir": f"logs_benchmark_{scheme}"},
    #         seeds=[0, 1, 2, 3, 4],
    #         in_dir="task_suites",
    #         suite_name="Final",
    #         sampled_json_path="task_suites/final_tasks.json",
    #     )