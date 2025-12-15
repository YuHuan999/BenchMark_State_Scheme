import gymnasium as gym
import numpy as np

class RepresentationWrapper(gym.Wrapper):
    """
    把 env 的 canonical obs 转成不同 representation。
    - scheme: "gate_seq" | "3d_tensor" | "graph" | ...
    - expose_mask: 是否把 env 的 action_mask 搬进 obs（推荐 True，但仍然是 env 的 mask）
    """
    def __init__(self, env, scheme: str, expose_mask: bool = True):
        super().__init__(env)
        self.scheme = scheme  ## "gate_seq" | "2d_grid" | "3d_tensor" | "graph"
        self.expose_mask = expose_mask
        if self.scheme == "gate_seq":
            # 你需要根据 scheme 设置 observation_space
            # 这里先用最通用的 Dict，state 的空间你自己按 scheme 填
            spaces = {"state": env.observation_space}  # 先占位：gate_seq
            if expose_mask:
                spaces["mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(spaces)

    def _to_representation(self, canonical_obs, info):
        """canonical_obs -> rep_state"""
        if self.scheme == "gate_seq":
            return canonical_obs

        elif self.scheme == "2d_grid":
            return self._build_2d_grid(canonical_obs, info)

        elif self.scheme == "3d_tensor":
            # 示例：你自己定义 encode 规则
            return self._build_3d_tensor(canonical_obs, info)

        elif self.scheme == "graph":
            # 示例：返回你 GNN 需要的数据结构（adj/node_feat/edge_index等）
            return self._build_graph(canonical_obs, info)

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def reset(self, **kwargs):
        canonical_obs, info = self.env.reset(**kwargs)

        state = self._to_representation(canonical_obs, info)
        if self.expose_mask:
            mask = info.get("action_mask", None)
            if mask is None:
                raise KeyError("env must provide info['action_mask'] when expose_mask=True")
            obs = {"state": state, "mask": mask.astype(np.int8)}
        else:
            obs = {"state": state}

        return obs, info

    def step(self, action):
        canonical_obs, reward, terminated, truncated, info = self.env.step(action)

        state = self._to_representation(canonical_obs, info)
        if self.expose_mask:
            mask = info.get("action_mask", None)
            if mask is None:
                raise KeyError("env must provide info['action_mask'] when expose_mask=True")
            obs = {"state": state, "mask": mask.astype(np.int8)}
        else:
            obs = {"state": state}

        return obs, reward, terminated, truncated, info

    # 下面两个你按你的编码方式实现
    def _build_2d_grid(self, canonical_obs, info):
        raise NotImplementedError

    def _build_3d_tensor(self, canonical_obs, info):
        raise NotImplementedError

    def _build_graph(self, canonical_obs, info):
        raise NotImplementedError
