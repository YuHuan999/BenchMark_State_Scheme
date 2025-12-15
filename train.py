import sys
sys.path.insert(0, r"E:\Projects\BenchMark_state_scheme\tianshou")

# import tianshou
# print("tianshou from:", tianshou.__file__)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams

from env import CircuitDesignerDiscrete
from prepare_tasks import generate_task_suite
from wrapper_represent import RepresentationWrapper
from encoders.encoder_MLP import Encoder_MLP

from qiskit import QuantumCircuit


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
             max_gates, max_depth, 
             seed=None, max_qubits=4, fidelity_threshold=0.99,
             success_reward=5.0, fail_reward=-5.0,
             render_mode=None, task_pool=None):

    env = CircuitDesignerDiscrete(
        max_qubits=max_qubits,
        max_gates=max_gates,
        max_depth=max_depth,
        task_pool=task_pool,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        seed=seed,
        render_mode=render_mode,
        mode=mode,
        fidelity_threshold=fidelity_threshold,  
        success_reward=success_reward,
        fail_reward=fail_reward,
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
        mask = obs.get("mask", None)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tasks
    # train_tasks, test_tasks = generate_task_suite(seed=0, n_tasks=60, mode="gates")
    # train_tasks = normalize_tasks(train_tasks)
    # test_tasks = normalize_tasks(test_tasks)

    ## 测试任务
    bell_circuit = QuantumCircuit(4)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    task = {"task_id": "bell",
            "qc": bell_circuit,
            "n_qubits": 4,
            "length_bin": "short",}
    train_tasks = [task]
    test_tasks = [task]


    scheme = "gate_seq"   # 你会用 argparse 切换
    n_train_env = 8
    n_test_env = 4

    train_envs = DummyVectorEnv([lambda: make_env(train_tasks, test_tasks, mode="train", scheme="gate_seq",
                                                  max_gates=5, max_depth=5,)] * n_train_env)
    test_envs  = DummyVectorEnv([lambda: make_env(train_tasks, test_tasks, mode="test",  scheme="gate_seq",
                                                  max_gates=5, max_depth=5,)] * n_test_env)

    # 从 env 拿动作维度
    dummy = make_env(train_tasks, test_tasks, mode="train", scheme="gate_seq",
                     max_gates=5, max_depth=5,)
    act_dim = int(dummy.action_space.n)

    # ---- 从 env 推断 max_gates（gate_seq 表示一般就是 state 的长度）----
    obs0, _ = dummy.reset()
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
        ent_coef=0.0,
        return_scaling=False,
        advantage_normalization=False,
        value_clip=False,
    )

    # on-policy buffer（大小 >= 每次收集的步数）
    buf = VectorReplayBuffer(total_size=20000, buffer_num=n_train_env)
    train_collector = Collector(algorithm, train_envs, buf, exploration_noise=False)
    test_collector  = Collector(algorithm, test_envs, exploration_noise=False)

    trainer_params = OnPolicyTrainerParams(
        training_collector=train_collector,
        test_collector=test_collector,
        max_epochs=20,
        epoch_num_steps=20000,
        update_step_num_repetitions=10,      # PPO更新轮数
        test_step_num_episodes=20,
        batch_size=256,
        collection_step_num_env_steps=2048,  # 每轮收集步数（按你env速度调）
        test_in_training=False,
    )
    result = algorithm.run_training(trainer_params)
    print(result)


if __name__ == "__main__":
    main()
