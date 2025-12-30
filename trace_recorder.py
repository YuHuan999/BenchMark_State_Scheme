import numpy as np
import torch
from torch.distributions import Categorical


def _to_batch(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def infer_step(ac, obs, device):
    """给定原始 obs(dict) 推理出 z/h/logits/value/mask（pre_action）。"""
    state = obs["state"]
    mask = obs.get("action_mask", None)

    state_t = _to_batch(state).to(device)
    mask_t = None
    if mask is not None:
        mask_t = _to_batch(mask).to(device)

    prev_mode = ac.training
    ac.eval()
    with torch.no_grad():
        z = ac.encoder(state_t)
        h = ac.shared(z)
        logits = ac.actor_head(h)
        if mask_t is not None:
            m = mask_t.bool()
            logits = logits.masked_fill(~m, -1e29)
        value = ac.critic_head(h).squeeze(-1)

    if prev_mode:
        ac.train()

    z_np = z.squeeze(0).detach().cpu().numpy()
    h_np = h.squeeze(0).detach().cpu().numpy()
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    value_np = float(value.squeeze(0).detach().cpu().item())
    if mask is None:
        mask_np = np.ones_like(logits_np, dtype=np.uint8)
    else:
        mask_np = np.array(mask, dtype=np.uint8).reshape(logits_np.shape)
    return z_np, h_np, logits_np, value_np, mask_np


def rollout_trace(env, ac, device, mode, episodes, seed_base, out_path):
    """采样轨迹并写压缩 npz，返回 meta 信息。"""
    env_id_list = []
    ep_id_list = []
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

    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            z_np, h_np, logits_np, value_np, mask_np = infer_step(ac, obs, device)

            if mode == "train":
                probs = torch.softmax(torch.from_numpy(logits_np), dim=-1)
                a = int(Categorical(probs=probs).sample().item())
            else:
                a = int(np.argmax(logits_np))

            obs, r, terminated, truncated, info = env.step(a)

            env_id_list.append(0)
            ep_id_list.append(ep)
            z_list.append(z_np.astype(np.float16))
            h_list.append(h_np.astype(np.float16))
            logits_list.append(logits_np.astype(np.float16))
            value_list.append(value_np)
            mask_list.append(mask_np.astype(np.uint8))
            act_list.append(a)
            fid_after_list.append(float(info.get("fidelity", 0.0)))
            rew_after_list.append(float(r))
            term_list.append(np.uint8(terminated))
            trunc_list.append(np.uint8(truncated))

            done = bool(terminated or truncated)

    def _stack(lst, dtype=None):
        arr = np.stack(lst, axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    out = {
        "env_id": _stack(env_id_list, np.int32),
        "episode_id": _stack(ep_id_list, np.int32),
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
    return {"episodes": episodes, "steps": len(act_list), "path": out_path}