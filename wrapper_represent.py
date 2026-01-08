import gymnasium as gym
import numpy as np

class RepresentationWrapper(gym.Wrapper):
    """
    把 env 的 canonical obs 转成不同 representation。
    - scheme: "gate_seq" | "3d_tensor" | "graph" | ...
    - expose_mask: 是否把 env 的 action_mask 搬进 obs（推荐 True，但仍然是 env 的 mask）
    
    3D-Tensor 方案说明（方案1）：
    ─────────────────────────────
    与 Patel/Kundu 的 "(N+3)×N slice" 的关键区别：
    - 方案1（本实现）：**按"动作序列 token"展开时间轴**，每个 token 占一个 t，
      表达"第 t 步做了什么 gate，在什么 qubit/role 上"。更贴合 env 的
      一步一个 action_id 的设计。
    - Patel/Kundu 类方案：通常按"电路 depth layer / moment"构 slice，并把
      CNOT 的 control/target 行 + 多种 1q gate 行拼成 (N+N1q)×N 的二维面，
      再堆成 Dmax 的 3D；它更偏"并行层/拓扑布局"的结构化电路描述。
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
                spaces["action_mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(spaces)
        
        elif self.scheme == "3d_tensor":
            # ========== 3D-Tensor scheme 初始化 ==========
            # 1) 获取环境参数
            self.n_qubits = getattr(env, "n_qubits", None) or getattr(env, "max_qubits", None)
            if self.n_qubits is None:
                raise ValueError("env 必须提供 n_qubits 或 max_qubits 属性")
            
            self.max_gates = getattr(env, "max_gates", None)
            if self.max_gates is None:
                # 尝试从 observation_space 推断
                if hasattr(env, "observation_space"):
                    obs_space = env.observation_space
                    if isinstance(obs_space, gym.spaces.Dict) and "state" in obs_space.spaces:
                        self.max_gates = obs_space["state"].shape[0]
                    elif hasattr(obs_space, "shape"):
                        self.max_gates = obs_space.shape[0]
            if self.max_gates is None:
                raise ValueError("无法推断 max_gates，env 需提供 max_gates 属性或 observation_space")
            
            # 2) 获取单量子比特门列表
            self.single_gates = getattr(env, "single_gates", None)
            if self.single_gates is None:
                # 从 env.gates 中去掉 CNOT
                all_gates = getattr(env, "gates", None)
                if all_gates is None:
                    raise ValueError("env 必须提供 single_gates 或 gates 属性")
                self.single_gates = [g for g in all_gates if g != "CNOT"]
            
            # 3) 构建 channel 映射
            # channels: [单比特门...] + [CNOT_control, CNOT_target]
            self.channel_names = list(self.single_gates) + ["CNOT_control", "CNOT_target"]
            self.channel_index = {name: idx for idx, name in enumerate(self.channel_names)}
            self.n_channels = len(self.channel_names)
            
            # 4) 设置 observation_space
            state_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.max_gates, self.n_qubits, self.n_channels),
                dtype=np.float32,
            )
            spaces = {"state": state_space}
            if expose_mask:
                spaces["action_mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(spaces)

        elif self.scheme == "2d_grid":
            # ========== 2D Grid scheme 初始化 ==========
            # 1) 获取环境参数
            self.n_qubits = getattr(env, "n_qubits", None) or getattr(env, "max_qubits", None)
            if self.n_qubits is None:
                raise ValueError("env 必须提供 n_qubits 或 max_qubits 属性")
            
            self.max_gates = getattr(env, "max_gates", None)
            if self.max_gates is None:
                # 尝试从 observation_space 推断
                if hasattr(env, "observation_space"):
                    obs_space = env.observation_space
                    if isinstance(obs_space, gym.spaces.Dict) and "state" in obs_space.spaces:
                        self.max_gates = obs_space["state"].shape[0]
                    elif hasattr(obs_space, "shape"):
                        self.max_gates = obs_space.shape[0]
            if self.max_gates is None:
                raise ValueError("无法推断 max_gates，env 需提供 max_gates 属性或 observation_space")
            
            # 2) 设置 padding 值和门 ID 映射
            self.PAD_Q = self.n_qubits  # sentinel value for padding
            self.oneq_gate_ids = {"H": 1, "X": 2, "Y": 3, "Z": 4, "T": 5}
            
            # 3) 设置 observation_space
            grid_space = gym.spaces.Box(
                low=0.0,
                high=float(max(self.PAD_Q, 5)),
                shape=(4, self.max_gates),
                dtype=np.float32,
            )
            spaces = {"state": grid_space}
            if expose_mask:
                spaces["action_mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(spaces)

        elif self.scheme == "graph":
            # ========== Graph (DAG) scheme 初始化 (GSQAS-style) ==========
            # 1) 获取环境参数
            self.n_qubits = getattr(env, "n_qubits", None) or getattr(env, "max_qubits", None)
            if self.n_qubits is None:
                raise ValueError("env 必须提供 n_qubits 或 max_qubits 属性")
            
            self.max_gates = getattr(env, "max_gates", None)
            if self.max_gates is None:
                # 尝试从 observation_space 推断
                if hasattr(env, "observation_space"):
                    obs_space = env.observation_space
                    if isinstance(obs_space, gym.spaces.Dict) and "state" in obs_space.spaces:
                        self.max_gates = obs_space["state"].shape[0]
                    elif hasattr(obs_space, "shape"):
                        self.max_gates = obs_space.shape[0]
            if self.max_gates is None:
                raise ValueError("无法推断 max_gates，env 需提供 max_gates 属性或 observation_space")
            
            # 2) 固定的 gate types（包括 START/END）
            self.graph_gate_types = ["H", "X", "Y", "Z", "T", "CNOT", "START", "END"]
            self.graph_gate_to_idx = {g: i for i, g in enumerate(self.graph_gate_types)}
            self.graph_K = 8  # gate type one-hot 维度
            
            # 3) 节点特征维度：K + 2*N (gate_type one-hot + control_mask + target_mask)
            self.graph_F = self.graph_K + 2 * self.n_qubits
            
            # 4) 最大节点数：max_gates + 2 (START + END)
            self.graph_Vmax = self.max_gates + 2
            
            # 5) 计算 state 向量的总维度
            # X_pad: (Vmax, F) -> Vmax * F
            # A_pad: (Vmax, Vmax) -> Vmax * Vmax
            # node_mask: (Vmax,) -> Vmax
            self.graph_state_dim = (
                self.graph_Vmax * self.graph_F +      # X_pad flattened
                self.graph_Vmax * self.graph_Vmax +   # A_pad flattened
                self.graph_Vmax                        # node_mask
            )
            
            # 6) 设置 observation_space
            state_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.graph_state_dim,),
                dtype=np.float32,
            )
            spaces = {"state": state_space}
            if expose_mask:
                spaces["action_mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(spaces)

    def _to_representation(self, canonical_obs, info):
        """canonical_obs -> rep_state"""
        # canonical_obs 可能是 dict({"state": ..., "action_mask": ...}) 或原始数组
        base_state = canonical_obs["state"] if isinstance(canonical_obs, dict) and "state" in canonical_obs else canonical_obs
        if self.scheme == "gate_seq":
            return base_state

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
            obs = {"state": state, "action_mask": mask.astype(np.int8)}
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
            obs = {"state": state, "action_mask": mask.astype(np.int8)}
        else:
            obs = {"state": state}

        return obs, reward, terminated, truncated, info

    # 下面几个你按你的编码方式实现
    def _build_2d_grid(self, canonical_obs, info):
        """
        将 canonical obs（action_id 序列，padding=-1）转换为 2D Grid 表示。
        
        输出 shape: (4, max_gates)
            行 0: cnot_control
            行 1: cnot_target
            行 2: oneq_target（单比特门作用 qubit）
            行 3: oneq_gate_id（单比特门类型 ID）
        
        编码规则：
            - CNOT(control=c, target=q)：grid[0,t]=c, grid[1,t]=q, grid[2,t]=PAD_Q, grid[3,t]=0
            - 单比特门 gate∈{H,X,Y,Z,T} 作用在 qubit=q：grid[0,t]=PAD_Q, grid[1,t]=PAD_Q, grid[2,t]=q, grid[3,t]=gate_id
            - padding=-1：grid[:,t] = (PAD_Q, PAD_Q, PAD_Q, 0)
        """
        # 提取 base state（action_id 序列）
        if isinstance(canonical_obs, dict) and "state" in canonical_obs:
            base_state = canonical_obs["state"]
        else:
            base_state = canonical_obs
        
        L = self.max_gates
        PAD_Q = self.PAD_Q
        
        # 初始化 grid，填充 padding 默认值
        grid = np.empty((4, L), dtype=np.float32)
        grid[0, :] = PAD_Q  # cnot_control
        grid[1, :] = PAD_Q  # cnot_target
        grid[2, :] = PAD_Q  # oneq_target
        grid[3, :] = 0      # oneq_gate_id
        
        # 获取 env 的 actions 映射表
        actions = getattr(self.env, "actions", None)
        if actions is None:
            actions = getattr(self.env.unwrapped, "actions", None)
        if actions is None:
            raise ValueError("无法获取 env.actions，请确保 env 提供 actions 属性")
        
        for t in range(L):
            aid = int(base_state[t])
            if aid < 0:
                # padding，保持默认值
                continue
            
            if aid >= len(actions):
                raise ValueError(
                    f"action_id={aid} 超出 env.actions 范围 [0, {len(actions)-1}]。"
                    f"请检查 state 与 action mapping 是否一致。"
                )
            
            a = actions[aid]
            g = a["gate"]
            
            if g == "CNOT":
                # 双量子比特门
                c = a.get("control", None)
                q = a.get("target", None)
                
                if c is None or q is None:
                    raise ValueError(
                        f"CNOT 门（action_id={aid}）缺少 control 或 target。"
                        f"action={a}"
                    )
                
                if not (0 <= c < self.n_qubits):
                    raise ValueError(
                        f"CNOT 门（action_id={aid}）control={c} 超出范围 [0, {self.n_qubits-1}]"
                    )
                if not (0 <= q < self.n_qubits):
                    raise ValueError(
                        f"CNOT 门（action_id={aid}）target={q} 超出范围 [0, {self.n_qubits-1}]"
                    )
                if c == q:
                    raise ValueError(
                        f"CNOT 门（action_id={aid}）control={c} 和 target={q} 相同。"
                    )
                    
                grid[0, t] = c  # cnot_control
                grid[1, t] = q  # cnot_target
                # grid[2,t] 和 grid[3,t] 保持 padding 默认值
            
            else:
                # 单量子比特门
                q = a.get("target", None)
                
                if q is None:
                    raise ValueError(
                        f"单比特门（action_id={aid}, gate={g}）缺少 target。"
                        f"action={a}"
                    )
                
                if not (0 <= q < self.n_qubits):
                    raise ValueError(
                        f"单比特门（action_id={aid}, gate={g}）target={q} 超出范围 [0, {self.n_qubits-1}]"
                    )
                
                if g not in self.oneq_gate_ids:
                    raise ValueError(
                        f"未知门类型 '{g}'（action_id={aid}）。"
                        f"当前支持的单比特门: {list(self.oneq_gate_ids.keys())}。"
                    )
                
                gid = self.oneq_gate_ids[g]
                # grid[0,t] 和 grid[1,t] 保持 padding 默认值
                grid[2, t] = q    # oneq_target
                grid[3, t] = gid  # oneq_gate_id
        
        return grid

    def _build_3d_tensor(self, canonical_obs, info):
        """
        将 canonical obs（action_id 序列，padding=-1）转换为 3D tensor。
        
        输出 shape: (D, N, C)
            D = self.max_gates（时间步/token 维度）
            N = self.n_qubits（量子比特维度）
            C = len(single_gates) + 2（channel 维度：各单比特门 + CNOT_control + CNOT_target）
        
        编码规则（binary 0/1）：
            - 单比特门 gate∈{H,X,Y,Z,T} 作用在 qubit=q：tensor[t, q, ch(gate)] = 1
            - CNOT(control=c, target=q)：tensor[t, c, ch("CNOT_control")] = 1
                                          tensor[t, q, ch("CNOT_target")] = 1
            - padding=-1 或超过 episode 长度：该 t 全 0
        """
        # 提取 base state（action_id 序列）
        if isinstance(canonical_obs, dict) and "state" in canonical_obs:
            base_state = canonical_obs["state"]
        else:
            base_state = canonical_obs
        
        D = self.max_gates
        N = self.n_qubits
        C = self.n_channels
        
        # 初始化全 0 tensor
        x = np.zeros((D, N, C), dtype=np.float32)
        
        # 获取 env 的 actions 映射表
        actions = self.env.actions
        
        for t in range(D):
            aid = int(base_state[t])
            if aid < 0:
                # padding，保持全 0
                continue
            
            if aid >= len(actions):
                raise ValueError(
                    f"action_id={aid} 超出 env.actions 范围 [0, {len(actions)-1}]。"
                    f"请检查 state 与 action mapping 是否一致。"
                )
            
            a = actions[aid]
            g = a["gate"]
            
            if g == "CNOT":
                # 双量子比特门：标记 control 和 target
                c = a["control"]
                q = a["target"]
                
                if c is None or q is None:
                    raise ValueError(
                        f"CNOT 门（action_id={aid}）缺少 control 或 target。"
                        f"action={a}"
                    )
                
                ctrl_ch = self.channel_index.get("CNOT_control")
                tgt_ch = self.channel_index.get("CNOT_target")
                
                if ctrl_ch is None or tgt_ch is None:
                    raise ValueError(
                        "channel_index 中缺少 'CNOT_control' 或 'CNOT_target'。"
                        f"当前 channel_names={self.channel_names}"
                    )
                
                x[t, c, ctrl_ch] = 1.0
                x[t, q, tgt_ch] = 1.0
            
            else:
                # 单量子比特门
                q = a["target"]
                
                if g not in self.channel_index:
                    raise ValueError(
                        f"未知门类型 '{g}'（action_id={aid}）。"
                        f"当前支持的单比特门: {self.single_gates}。"
                        f"请检查 env.single_gates 是否与 action mapping 一致。"
                    )
                
                ch = self.channel_index[g]
                x[t, q, ch] = 1.0
        
        return x

    def _build_graph(self, canonical_obs, info):
        """
        将 canonical obs（action_id 序列，padding=-1）转换为 Graph (DAG) 表示。
        
        输出：flat state_vec，shape = (graph_state_dim,)，dtype = float32
        
        内部结构（拼接）：
            - X_pad:     节点特征矩阵 (Vmax, F) -> flatten
            - A_pad:     邻接矩阵 (Vmax, Vmax) -> flatten
            - node_mask: 有效节点 mask (Vmax,)
        
        节点顺序：[START] + gates_in_prefix + [END]
        
        节点特征 F = K + 2*N：
            - gate type one-hot (K=8): H,X,Y,Z,T,CNOT,START,END
            - control_mask (N): CNOT 的 control qubit 位置
            - target_mask (N): 单比特门的 target 或 CNOT 的 target
        
        边：按每条 qubit 线上的时序依赖连 DAG（last_touch 方式）
        """
        # 提取 base state（action_id 序列）
        if isinstance(canonical_obs, dict) and "state" in canonical_obs:
            base_state = canonical_obs["state"]
        else:
            base_state = canonical_obs
        
        N = self.n_qubits
        K = self.graph_K
        F = self.graph_F
        Vmax = self.graph_Vmax
        
        # 获取有效的 action_ids（非 padding）
        valid_aids = [int(aid) for aid in base_state if int(aid) >= 0]
        L = len(valid_aids)  # 有效 gate 数
        V = L + 2            # 总节点数（含 START/END）
        
        # 获取 env 的 actions 映射表
        actions = getattr(self.env, "actions", None)
        if actions is None:
            actions = getattr(self.env.unwrapped, "actions", None)
        if actions is None:
            raise ValueError("无法获取 env.actions，请确保 env 提供 actions 属性")
        
        # 初始化
        X_pad = np.zeros((Vmax, F), dtype=np.float32)
        A_pad = np.zeros((Vmax, Vmax), dtype=np.float32)
        node_mask = np.zeros((Vmax,), dtype=np.float32)
        
        # 节点索引
        start_idx = 0
        end_idx = V - 1  # L + 1
        
        # ===== 1) 填充节点特征 =====
        
        # START 节点（idx=0）
        X_pad[start_idx, self.graph_gate_to_idx["START"]] = 1.0
        node_mask[start_idx] = 1.0
        
        # END 节点（idx=V-1）
        X_pad[end_idx, self.graph_gate_to_idx["END"]] = 1.0
        node_mask[end_idx] = 1.0
        
        # Gate 节点（idx=1..L）
        for i, aid in enumerate(valid_aids):
            node_idx = i + 1  # gate 节点从 idx=1 开始
            
            if aid >= len(actions):
                raise ValueError(
                    f"action_id={aid} 超出 env.actions 范围 [0, {len(actions)-1}]"
                )
            
            a = actions[aid]
            g = a["gate"]
            target = a["target"]
            control = a.get("control", None)
            
            # gate type one-hot
            if g in self.graph_gate_to_idx:
                X_pad[node_idx, self.graph_gate_to_idx[g]] = 1.0
            else:
                raise ValueError(f"未知门类型 '{g}'，支持的类型: {self.graph_gate_types}")
            
            # qubit masks
            if g == "CNOT":
                # CNOT: control_mask[control]=1, target_mask[target]=1
                if control is not None:
                    X_pad[node_idx, K + control] = 1.0           # control_mask
                if target is not None:
                    X_pad[node_idx, K + N + target] = 1.0        # target_mask
            else:
                # 单比特门：只有 target_mask[target]=1
                if target is not None:
                    X_pad[node_idx, K + N + target] = 1.0        # target_mask
            
            node_mask[node_idx] = 1.0
        
        # ===== 2) 构建边（邻接矩阵）=====
        # 按每条 qubit 线上的时序依赖连 DAG
        
        # last_touch[q] = 上一个访问 qubit q 的节点 idx
        last_touch = {q: start_idx for q in range(N)}
        
        # 对每个 gate 节点
        for i, aid in enumerate(valid_aids):
            node_idx = i + 1
            
            a = actions[aid]
            g = a["gate"]
            target = a["target"]
            control = a.get("control", None)
            
            # 找该 gate 涉及的 qubits
            if g == "CNOT":
                involved_qubits = []
                if control is not None:
                    involved_qubits.append(control)
                if target is not None:
                    involved_qubits.append(target)
            else:
                involved_qubits = [target] if target is not None else []
            
            # 对每个涉及的 qubit，加边 last_touch[q] -> node_idx
            for q in involved_qubits:
                prev_idx = last_touch[q]
                if prev_idx != node_idx:  # 不加自环
                    A_pad[prev_idx, node_idx] = 1.0
                last_touch[q] = node_idx
        
        # 最后对每个 qubit，加边 last_touch[q] -> end_idx
        for q in range(N):
            prev_idx = last_touch[q]
            if prev_idx != end_idx:  # 不加自环
                A_pad[prev_idx, end_idx] = 1.0
        
        # ===== 3) 拼接成 flat state_vec =====
        state_vec = np.concatenate([
            X_pad.flatten(),
            A_pad.flatten(),
            node_mask
        ]).astype(np.float32)
        
        return state_vec

    def unpack_graph_state(self, state_vec):
        """
        将 flat state_vec 解包为 (X, A, node_mask)。
        
        Args:
            state_vec: shape = (graph_state_dim,), dtype = float32
        
        Returns:
            X:         节点特征矩阵, shape = (Vmax, F)
            A:         邻接矩阵, shape = (Vmax, Vmax)
            node_mask: 有效节点 mask, shape = (Vmax,)
        """
        if self.scheme != "graph":
            raise ValueError("unpack_graph_state 仅适用于 scheme='graph'")
        
        Vmax = self.graph_Vmax
        F = self.graph_F
        
        # 计算切分点
        x_size = Vmax * F
        a_size = Vmax * Vmax
        
        # 切片并 reshape
        X = state_vec[:x_size].reshape(Vmax, F)
        A = state_vec[x_size:x_size + a_size].reshape(Vmax, Vmax)
        node_mask = state_vec[x_size + a_size:]
        
        return X, A, node_mask


# ===================== Self-Test =====================
if __name__ == "__main__":
    """
    测试考虑些在别的文件中使用
    Minimal self-test: verify basic functionality of 3d_tensor scheme.
    
    Run:
        python wrapper_represent.py
    
    Note: requires env.py in the same directory.
    """
    from qiskit import QuantumCircuit
    
    # Try to import env
    try:
        from env import CircuitDesignerDiscrete
    except ImportError:
        print("Cannot import CircuitDesignerDiscrete, ensure env.py is in the same directory.")
        exit(1)
    
    # Create a minimal task pool (2-qubit simple circuits)
    def make_simple_task_pool():
        tasks = []
        # Task 1: simple H gate
        qc1 = QuantumCircuit(4)
        qc1.h(0)
        qc1.x(1)
        tasks.append({"task_id": "test_1", "qc": qc1, "n_qubits": 2})
        
        # Task 2: CNOT
        qc2 = QuantumCircuit(4)
        qc2.h(0)
        qc2.cx(0, 1)
        tasks.append({"task_id": "test_2", "qc": qc2, "n_qubits": 2})
        
        # Task 3: multi-gate combination
        qc3 = QuantumCircuit(4)
        qc3.h(0)
        qc3.cx(0, 1)
        qc3.z(1)
        qc3.t(0)
        tasks.append({"task_id": "test_3", "qc": qc3, "n_qubits": 2})
        
        return tasks
    
    print("=" * 60)
    print("3D-Tensor Scheme Self-Test")
    print("=" * 60)
    
    # 创建环境
    task_pool = make_simple_task_pool()
    base_env = CircuitDesignerDiscrete(
        task_pool=task_pool,
        max_qubits=4,
        max_gates=20,  # 较小值方便测试
    )
    
    # 用 RepresentationWrapper 包装
    env = RepresentationWrapper(base_env, scheme="3d_tensor", expose_mask=True)
    
    # Check observation_space
    print(f"\n[1] observation_space check:")
    print(f"    env.observation_space = {env.observation_space}")
    print(f"    env.n_qubits = {env.n_qubits}")
    print(f"    env.max_gates = {env.max_gates}")
    print(f"    env.n_channels = {env.n_channels}")
    print(f"    env.channel_names = {env.channel_names}")
    
    expected_shape = (env.max_gates, env.n_qubits, env.n_channels)
    print(f"    expected state shape = {expected_shape}")
    
    # Reset test
    print(f"\n[2] reset() test:")
    obs, info = env.reset(seed=42)
    
    state = obs["state"]
    print(f"    obs['state'].shape = {state.shape}")
    print(f"    obs['state'].dtype = {state.dtype}")
    print(f"    obs['state'].min() = {state.min()}")
    print(f"    obs['state'].max() = {state.max()}")
    
    # Assertions
    assert state.ndim == 3, f"state should be 3D, got ndim={state.ndim}"
    assert state.shape == expected_shape, f"shape mismatch: {state.shape} vs {expected_shape}"
    assert state.dtype == np.float32, f"dtype should be float32, got {state.dtype}"
    assert state.min() >= 0.0, f"min should be >= 0, got {state.min()}"
    assert state.max() <= 1.0, f"max should be <= 1, got {state.max()}"
    print("    [OK] reset() check passed")
    
    # Check action_mask
    if "action_mask" in obs:
        mask = obs["action_mask"]
        print(f"    obs['action_mask'].shape = {mask.shape}")
        print(f"    valid actions = {mask.sum()} / {len(mask)}")
    
    # Step test
    print(f"\n[3] step() test:")
    mask = obs["action_mask"]
    legal_actions = np.where(mask == 1)[0]
    if len(legal_actions) == 0:
        print("    Warning: no legal actions available")
    else:
        action = np.random.choice(legal_actions)
        action_info = env.env.get_action_info(action)
        print(f"    selected action: {action} -> {action_info}")
        
        obs2, reward, terminated, truncated, info2 = env.step(action)
        state2 = obs2["state"]
        
        print(f"    obs['state'].shape = {state2.shape}")
        print(f"    obs['state'].dtype = {state2.dtype}")
        print(f"    reward = {reward}")
        print(f"    terminated = {terminated}, truncated = {truncated}")
        
        # Assertions
        assert state2.ndim == 3, f"after step state should be 3D, got ndim={state2.ndim}"
        assert state2.shape == expected_shape, f"after step shape mismatch: {state2.shape} vs {expected_shape}"
        assert state2.dtype == np.float32, f"after step dtype should be float32, got {state2.dtype}"
        assert state2.min() >= 0.0, f"after step min should be >= 0, got {state2.min()}"
        assert state2.max() <= 1.0, f"after step max should be <= 1, got {state2.max()}"
        
        # Check if first step is correctly encoded
        if state2.max() > 0:
            # Find non-zero positions
            nonzero = np.argwhere(state2 > 0)
            print(f"    non-zero positions (t, q, ch): {nonzero.tolist()}")
        
        print("    [OK] step() check passed")
    
    # Multi-step test
    print(f"\n[4] multi-step test (run 5 steps or until done):")
    obs, info = env.reset(seed=123)
    for step_i in range(5):
        mask = obs["action_mask"]
        legal_actions = np.where(mask == 1)[0]
        if len(legal_actions) == 0:
            print(f"    step {step_i}: no legal actions, stopping")
            break
        
        action = np.random.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = obs["state"]
        nonzero_count = np.sum(state > 0)
        print(f"    step {step_i}: action={action}, non-zero count={nonzero_count}, reward={reward:.4f}")
        
        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print("\n" + "=" * 60)
    print("[OK] 3D-Tensor self-tests passed!")
    print("=" * 60)
    
    # ===================== Graph Scheme Self-Test =====================
    print("\n")
    print("=" * 60)
    print("Graph (DAG) Scheme Self-Test")
    print("=" * 60)
    
    # 创建新的 base_env（重新创建以保证干净状态）
    base_env_g = CircuitDesignerDiscrete(
        task_pool=task_pool,
        max_qubits=4,
        max_gates=20,
    )
    
    # 用 RepresentationWrapper 包装为 graph scheme
    env_g = RepresentationWrapper(base_env_g, scheme="graph", expose_mask=True)
    
    # Check observation_space
    print(f"\n[1] observation_space check:")
    print(f"    env_g.observation_space = {env_g.observation_space}")
    print(f"    env_g.n_qubits = {env_g.n_qubits}")
    print(f"    env_g.max_gates = {env_g.max_gates}")
    print(f"    env_g.graph_K = {env_g.graph_K}")
    print(f"    env_g.graph_F = {env_g.graph_F}")
    print(f"    env_g.graph_Vmax = {env_g.graph_Vmax}")
    print(f"    env_g.graph_state_dim = {env_g.graph_state_dim}")
    
    expected_dim = env_g.graph_state_dim
    print(f"    expected state dim = {expected_dim}")
    
    # Reset test (empty circuit)
    print(f"\n[2] reset() test (empty circuit, L=0):")
    obs_g, info_g = env_g.reset(seed=0)
    
    state_g = obs_g["state"]
    print(f"    obs['state'].ndim = {state_g.ndim}")
    print(f"    obs['state'].shape = {state_g.shape}")
    print(f"    obs['state'].dtype = {state_g.dtype}")
    print(f"    obs['state'].min() = {state_g.min()}")
    print(f"    obs['state'].max() = {state_g.max()}")
    
    # Assertions
    assert state_g.ndim == 1, f"graph state should be 1D, got ndim={state_g.ndim}"
    assert state_g.shape[0] == expected_dim, f"shape mismatch: {state_g.shape[0]} vs {expected_dim}"
    assert state_g.dtype == np.float32, f"dtype should be float32, got {state_g.dtype}"
    assert state_g.min() >= 0.0, f"min should be >= 0, got {state_g.min()}"
    assert state_g.max() <= 1.0, f"max should be <= 1, got {state_g.max()}"
    print("    [OK] reset() check passed")
    
    # 测试 unpack_graph_state
    print(f"\n[3] unpack_graph_state() test:")
    X, A, node_mask = env_g.unpack_graph_state(state_g)
    print(f"    X.shape = {X.shape} (expected: ({env_g.graph_Vmax}, {env_g.graph_F}))")
    print(f"    A.shape = {A.shape} (expected: ({env_g.graph_Vmax}, {env_g.graph_Vmax}))")
    print(f"    node_mask.shape = {node_mask.shape} (expected: ({env_g.graph_Vmax},))")
    
    assert X.shape == (env_g.graph_Vmax, env_g.graph_F), f"X shape mismatch"
    assert A.shape == (env_g.graph_Vmax, env_g.graph_Vmax), f"A shape mismatch"
    assert node_mask.shape == (env_g.graph_Vmax,), f"node_mask shape mismatch"
    
    # Empty circuit should have only START and END nodes valid
    valid_nodes = int(node_mask.sum())
    print(f"    valid nodes (should be 2 when L=0): {valid_nodes}")
    assert valid_nodes == 2, f"Empty circuit should have 2 valid nodes (START+END), got {valid_nodes}"
    assert node_mask[0] == 1.0, "START node (idx=0) should be valid"
    assert node_mask[1] == 1.0, "END node (idx=1, since L=0) should be valid"
    
    # Check START->END edge: for each qubit there should be a START->END edge
    # In empty circuit, A[0,1] should = 1 (all qubits have START->END)
    print(f"    A[START, END] = A[0, 1] = {A[0, 1]} (should be 1.0)")
    assert A[0, 1] == 1.0, f"Empty circuit should have START->END edge, got A[0,1]={A[0,1]}"
    print("    [OK] unpack_graph_state() check passed")
    
    # Check action_mask
    if "action_mask" in obs_g:
        mask_g = obs_g["action_mask"]
        print(f"\n    obs['action_mask'].shape = {mask_g.shape}")
        print(f"    valid actions = {mask_g.sum()} / {len(mask_g)}")
    
    # Step test
    print(f"\n[4] step() test:")
    mask_g = obs_g["action_mask"]
    legal_actions_g = np.where(mask_g == 1)[0]
    if len(legal_actions_g) == 0:
        print("    Warning: no legal actions available")
    else:
        action_g = np.random.choice(legal_actions_g)
        action_info_g = env_g.env.get_action_info(action_g)
        print(f"    selected action: {action_g} -> {action_info_g}")
        
        obs_g2, reward_g, terminated_g, truncated_g, info_g2 = env_g.step(action_g)
        state_g2 = obs_g2["state"]
        
        print(f"    obs['state'].ndim = {state_g2.ndim}")
        print(f"    obs['state'].shape = {state_g2.shape}")
        print(f"    obs['state'].dtype = {state_g2.dtype}")
        print(f"    reward = {reward_g}")
        print(f"    terminated = {terminated_g}, truncated = {truncated_g}")
        
        # Assertions
        assert state_g2.ndim == 1, f"after step state should be 1D, got ndim={state_g2.ndim}"
        assert state_g2.shape[0] == expected_dim, f"after step shape mismatch: {state_g2.shape[0]} vs {expected_dim}"
        assert state_g2.dtype == np.float32, f"after step dtype should be float32, got {state_g2.dtype}"
        assert state_g2.min() >= 0.0, f"after step min should be >= 0, got {state_g2.min()}"
        assert state_g2.max() <= 1.0, f"after step max should be <= 1, got {state_g2.max()}"
        
        # Unpack and check
        X2, A2, node_mask2 = env_g.unpack_graph_state(state_g2)
        valid_nodes2 = int(node_mask2.sum())
        print(f"    valid nodes after step (should be 3): {valid_nodes2}")
        assert valid_nodes2 == 3, f"After one step should have 3 valid nodes, got {valid_nodes2}"
        
        print("    [OK] step() check passed")
    
    # Multi-step test
    print(f"\n[5] multi-step test (run 5 steps or until done):")
    obs_g, info_g = env_g.reset(seed=123)
    for step_i in range(5):
        mask_g = obs_g["action_mask"]
        legal_actions_g = np.where(mask_g == 1)[0]
        if len(legal_actions_g) == 0:
            print(f"    step {step_i}: no legal actions, stopping")
            break
        
        action_g = np.random.choice(legal_actions_g)
        obs_g, reward_g, terminated_g, truncated_g, info_g = env_g.step(action_g)
        
        state_g = obs_g["state"]
        X, A, node_mask = env_g.unpack_graph_state(state_g)
        valid_nodes = int(node_mask.sum())
        edge_count = int(A.sum())
        print(f"    step {step_i}: action={action_g}, valid_nodes={valid_nodes}, edges={edge_count}, reward={reward_g:.4f}")
        
        if terminated_g or truncated_g:
            print(f"    Episode ended: terminated={terminated_g}, truncated={truncated_g}")
            break
    
    print("\n" + "=" * 60)
    print("[OK] Graph self-tests passed!")
    print("=" * 60)
    
    # ===================== 2D Grid Scheme Self-Test =====================
    print("\n")
    print("=" * 60)
    print("2D Grid Scheme Self-Test")
    print("=" * 60)
    
    # 创建新的 base_env
    base_env_2d = CircuitDesignerDiscrete(
        task_pool=task_pool,
        max_qubits=4,
        max_gates=20,
    )
    
    # 用 RepresentationWrapper 包装为 2d_grid scheme
    env2 = RepresentationWrapper(base_env_2d, scheme="2d_grid", expose_mask=True)
    
    # Check observation_space
    print(f"\n[1] observation_space check:")
    print(f"    env2.observation_space = {env2.observation_space}")
    print(f"    env2.n_qubits = {env2.n_qubits}")
    print(f"    env2.max_gates = {env2.max_gates}")
    print(f"    env2.PAD_Q = {env2.PAD_Q}")
    print(f"    env2.oneq_gate_ids = {env2.oneq_gate_ids}")
    
    expected_shape_2d = (4, env2.max_gates)
    print(f"    expected state shape = {expected_shape_2d}")
    
    # Reset test
    print(f"\n[2] reset() test:")
    obs2, info2 = env2.reset(seed=0)
    
    state_2d = obs2["state"]
    print(f"    obs['state'].shape = {state_2d.shape}")
    print(f"    obs['state'].dtype = {state_2d.dtype}")
    print(f"    obs['state'].min() = {state_2d.min()}")
    print(f"    obs['state'].max() = {state_2d.max()}")
    
    # Assertions for reset
    assert state_2d.shape == expected_shape_2d, f"shape mismatch: {state_2d.shape} vs {expected_shape_2d}"
    assert state_2d.dtype == np.float32, f"dtype should be float32, got {state_2d.dtype}"
    assert state_2d.min() >= 0, f"min should be >= 0, got {state_2d.min()}"
    assert state_2d[0, :].max() <= env2.PAD_Q, f"cnot_control max should be <= PAD_Q={env2.PAD_Q}, got {state_2d[0, :].max()}"
    assert state_2d[1, :].max() <= env2.PAD_Q, f"cnot_target max should be <= PAD_Q={env2.PAD_Q}, got {state_2d[1, :].max()}"
    assert state_2d[2, :].max() <= env2.PAD_Q, f"oneq_target max should be <= PAD_Q={env2.PAD_Q}, got {state_2d[2, :].max()}"
    assert state_2d[3, :].max() <= 5, f"oneq_gate_id max should be <= 5, got {state_2d[3, :].max()}"
    print("    [OK] reset() check passed")
    
    # Check action_mask
    if "action_mask" in obs2:
        mask_2d = obs2["action_mask"]
        print(f"    obs['action_mask'].shape = {mask_2d.shape}")
        print(f"    valid actions = {mask_2d.sum()} / {len(mask_2d)}")
    
    # Multi-step test with assertions
    print(f"\n[3] multi-step test (run 1~3 steps):")
    obs2, info2 = env2.reset(seed=0)
    for step_i in range(3):
        mask_2d = obs2["action_mask"]
        legal_actions_2d = np.where(mask_2d == 1)[0]
        if len(legal_actions_2d) == 0:
            print(f"    step {step_i}: no legal actions, stopping")
            break
        
        action_2d = np.random.choice(legal_actions_2d)
        action_info_2d = env2.env.get_action_info(action_2d)
        print(f"    step {step_i}: action={action_2d} -> {action_info_2d}")
        
        obs2, reward_2d, terminated_2d, truncated_2d, info2 = env2.step(action_2d)
        state_2d = obs2["state"]
        
        # Assertions after each step
        assert state_2d.shape == expected_shape_2d, f"step {step_i}: shape mismatch"
        assert state_2d.dtype == np.float32, f"step {step_i}: dtype mismatch"
        assert state_2d.min() >= 0, f"step {step_i}: min should be >= 0"
        assert state_2d[0, :].max() <= env2.PAD_Q, f"step {step_i}: cnot_control exceeds PAD_Q"
        assert state_2d[1, :].max() <= env2.PAD_Q, f"step {step_i}: cnot_target exceeds PAD_Q"
        assert state_2d[2, :].max() <= env2.PAD_Q, f"step {step_i}: oneq_target exceeds PAD_Q"
        assert state_2d[3, :].max() <= 5, f"step {step_i}: oneq_gate_id exceeds 5"
        
        print(f"           state shape={state_2d.shape}, reward={reward_2d:.4f}")
        
        if terminated_2d or truncated_2d:
            print(f"    Episode ended: terminated={terminated_2d}, truncated={truncated_2d}")
            break
    
    print("    [OK] multi-step check passed")
    
    # Verify padding columns
    print(f"\n[4] padding verification:")
    obs2, info2 = env2.reset(seed=0)
    state_2d = obs2["state"]
    # At reset, all columns should be padding (no gates yet)
    for t in range(env2.max_gates):
        assert state_2d[0, t] == env2.PAD_Q, f"column {t}: cnot_control should be PAD_Q"
        assert state_2d[1, t] == env2.PAD_Q, f"column {t}: cnot_target should be PAD_Q"
        assert state_2d[2, t] == env2.PAD_Q, f"column {t}: oneq_target should be PAD_Q"
        assert state_2d[3, t] == 0, f"column {t}: oneq_gate_id should be 0"
    print("    [OK] all columns are properly padded at reset")
    
    print("\n" + "=" * 60)
    print("[OK] All self-tests passed!")
    print("=" * 60)
