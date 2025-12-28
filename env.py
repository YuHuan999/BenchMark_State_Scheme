import numpy as np
import gymnasium as gym
import time
from gymnasium.envs.registration import register

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, TGate, CXGate
from qiskit.quantum_info import Operator


import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)


class CircuitDesignerDiscrete(gym.Env):
    """离散动作空间的量子电路重构环境（仅重建目标电路）。"""

    metadata = {"render_modes": ["image", "text"], "render_fps": 30}

    def __init__(
        self,
        
        task_pool=None,
        train_tasks=None,
        test_tasks=None,
        seed=None,
        render_mode=None,
        mode='train',
        max_qubits=4,   # 统一 one-hot 编码
        # max_gates=10,   # 统一神经网络输入维度
        max_gates=78,   # 统一神经网络输入维度
        max_depth = 78,
        fidelity_threshold: float = 0.99,
        success_reward: float = 6.0,
        fail_reward: float = -5.0,
        gates_penalty: float = 1.0,
    ):
        super().__init__()
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # 兼容：优先使用 train_tasks/test_tasks，否则回退 task_pool（同时作为训练和测试）
        if (train_tasks is None or test_tasks is None) and task_pool is None:
            raise ValueError("需提供 train_tasks/test_tasks 或 task_pool。")
        self.train_tasks = train_tasks if train_tasks is not None else task_pool
        self.test_tasks = test_tasks if test_tasks is not None else task_pool
        if not self.train_tasks or not self.test_tasks:
            raise ValueError("任务列表不能为空。")

        self.render_mode = render_mode
        self.mode = mode
        self.name = f"REBUILD|{max_qubits}-g{max_depth}"

        self.max_qubits = max_qubits
        self.n_qubits = max_qubits
        self.max_gates = max_gates ## max_gates <= max_depth * n_qubits
        self.max_depth = max_depth
        self.fidelity_threshold = fidelity_threshold
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.gates_penalty = gates_penalty
        self.profile = {
            "env_step_time": 0.0,
            "env_step_calls": 0,
            "illegal_count": 0,
            "mask_valid_sum": 0.0,
            "mask_calls": 0,
        }
        

        self._qc = QuantumCircuit(self.n_qubits)
        self.qc_operator = Operator(self._qc) 
        self.current_depth = self._qc.depth()
        self.current_gates = len(self._qc.data)
        self.task_budget = 0 ## 实际在训练中可查入门的上限
        self.target_qc = None
        self.target_unitary = None
        self.gates_target = 0
        self.current_task_id = None
        # 任务难度档位（兼容旧的 length_bin，现统一使用 difficulty_bin）
        self.current_difficulty_bin = None
        self.gate_tokens = []
        self._qubit_stacks = [[] for _ in range(self.n_qubits)]
        self.fidelities = []

        # 预计算局部酉矩阵（用于 redundancy 判定）
        self._localU_1q, self._localU_2q = self._build_local_unitaries()
        self._eps_comm = 1e-8
        self._eps_gate = 1e-8

        # 定义门类型
        self.gates = ['H', 'X', 'Y', 'Z', 'T', 'CNOT']
        self.single_gates = ['H', 'X', 'Y', 'Z', 'T']  # 单量子比特门
        self.two_gates = ['CNOT']  # 双量子比特门

        # 离散动作空间：为每种可能的门-量子比特组合分配一个动作ID
        self.actions = self._create_action_mapping()
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # 观察空间：定长门序列，padding=-1
        state_space = gym.spaces.Box(
            low=-1,
            high=len(self.actions) - 1,
            shape=(self.max_gates,),
            dtype=np.int32,
        )
        mask_space = gym.spaces.MultiBinary(len(self.actions))
        self.observation_space = gym.spaces.Dict({"state": state_space, "action_mask": mask_space})

        print(f"Created {len(self.actions)} discrete actions")

    def _create_action_mapping(self):
        """创建动作ID到(门类型, 量子比特)的映射"""
        actions = []

        # 单量子比特门：每个门在每个量子比特上
        for gate in self.single_gates:
            for qubit in range(self.n_qubits):
                actions.append({'gate': gate, 'target': qubit, 'control': None})

        # 双量子比特门：每个门在每对不同量子比特上
        for gate in self.two_gates:
            for control in range(self.n_qubits):
                for target in range(self.n_qubits):
                    if control != target:  # 控制和目标量子比特必须不同
                        actions.append({'gate': gate, 'target': target, 'control': control})

        return actions

    def _build_local_unitaries(self):
        """构建局部酉矩阵缓存，避免在 mask 中重复计算。"""
        u1 = {}
        for gate_name, gate_cls in [('H', HGate), ('X', XGate), ('Y', YGate), ('Z', ZGate), ('T', TGate)]:
            qc = QuantumCircuit(1)
            qc.append(gate_cls(), [0])
            u1[gate_name] = Operator(qc).data

        qc2 = QuantumCircuit(2)
        qc2.append(CXGate(), [0, 1])
        u2 = {'CNOT': Operator(qc2).data}
        return u1, u2

    def _parse_task(self, task):

        active = task["n_qubits"]
        for inst, qargs, cargs in task["qc"].data:
            for q in qargs:
                if q._index >= active:   # qiskit qubit index
                    raise ValueError(f"Found op on idle qubit {q._index} for active={active}, task={task['task_id']}")
                
        assert task["qc"].num_qubits == 4, f"target qc should be 4-qubit, got {task['qc'].num_qubits}"
        assert 2 <= task["n_qubits"] <= 4
        
        """解析任务条目，返回 (task_id, target_qc, difficulty_bin)。"""
        task_id = None
        target_qc = None
        difficulty_bin = None
        gates_count = None


        if isinstance(task, dict):
            task_id = task.get('id') or task.get('task_id')
            target_qc = task.get('qc') or task.get('circuit') or task.get('target')
            # 新字段 difficulty_bin 优先，兼容旧的 length_bin
            difficulty_bin = task.get('difficulty_bin', task.get('length_bin'))
            gates_count = len(target_qc.data)
        elif isinstance(task, (tuple, list)) and len(task) == 2:
            task_id, target_qc = task
            gates_count = len(target_qc.data)
        if target_qc is None:
            raise ValueError("task_pool 中的任务需包含 QuantumCircuit 实例。")
        if task_id is None:
            task_id = f"task_{id(target_qc)}"

        if not isinstance(target_qc, QuantumCircuit):
            raise TypeError("task_pool 中的目标需为 QuantumCircuit。")

        return str(task_id), target_qc, difficulty_bin, gates_count, int(active)

    def _sample_task(self):
        """从任务池中随机抽取一个任务"""
        task_list = self.train_tasks if self.mode == 'train' else self.test_tasks
        idx = self._np_random.choice(len(task_list))
        return self._parse_task(task_list[idx])

    def _operation(self, action_id):
        """将动作ID转换为量子门操作"""
        if action_id >= len(self.actions):
            return None

        action = self.actions[action_id]
        gate_type = action['gate']
        target = action['target']
        control = action['control']

        # 单量子比特门
        if gate_type == 'H':
            return HGate(), [target]
        if gate_type == 'X':
            return XGate(), [target]
        if gate_type == 'Y':
            return YGate(), [target]
        if gate_type == 'Z':
            return ZGate(), [target]
        if gate_type == 'T':
            return TGate(), [target]

        # 双量子比特门
        if gate_type == 'CNOT':
            return CXGate(), [control, target]

        return None

    def _compute_fidelity(self):
        """计算当前电路酉阵与目标电路酉阵的保真度（Hilbert–Schmidt 内积）。"""
        if self.target_unitary is None:
            return 0.0
        U_current = self.qc_operator.data
        U_target = self.target_unitary
        if U_current.shape != U_target.shape:
            raise ValueError(
            f"Unitary shape mismatch: U_current{U_current.shape} vs U_target{U_target.shape}. "
            f"可能是 n_qubits 不一致，或 reset 后 actions/qc 未正确重建。" )
        hs_inner = np.trace(U_current.conj().T @ U_target) / U_target.shape[0]
        fidelity = float(np.abs(hs_inner) ** 2)
        return fidelity

    def reward(self, fidelity: float) -> float:
        """
        - 若 fidelity >= self.fidelity_threshold(τ)：奖励 = self.success_reward(R)
        - 否则：奖励 = fidelity - self.fidelity_threshold
        """
        tau = float(self.fidelity_threshold)
        R = float(self.success_reward)
        F = float(fidelity)
        return float(R if F >= tau else (F - tau))

    def reward_cost(self, fidelity: float, 
                    prev_fidelity: float, 
                    current_depth: int, 
                    current_gates: int) -> float:
        """
        成本奖励（cost = 1 - fidelity），以“达到阈值 tau”为目标，深度 max_depth 为预算上限：
        - 成功：F_t >= tau -> R_succ
        - 失败：depth >= max_depth 且仍未成功 -> R_fail
        - 否则：max( (C_prev - C_t) / (C_prev - E_min), -1 )
        其中 E_min = 1 - tau（以“刚好达标”的cost作为参考下界）
        """
        tau = float(self.fidelity_threshold)
        E_min = 1.0 - tau
        R_succ = float(self.success_reward) ## 6.0
        R_fail = float(self.fail_reward)  ## -5.0
        lam_len = float(self.gates_penalty) ## 1.0
        F_t = float(fidelity)
        F_prev = float(prev_fidelity)
        
        target_gates = len(self.target_qc.data)

        # 1) 成功（更直观）
        if F_t >= tau:
            current_gates = self.current_gates
            return R_succ - (current_gates / target_gates) * lam_len
        # # 2) 失败（预算用深度）
        # if int(current_depth) >= int(self.max_depth):
        #     return R_fail
        # 2) 失败（预算用门数）
        if int(current_gates) >= int(self.task_budget):
            return R_fail
        # 3) 过程奖励：用 cost 改进做归一化
        C_t = 1.0 - F_t
        C_prev = 1.0 - F_prev
        denom = (C_prev - E_min)
        if abs(denom) < 1e-8:   # 比 1e-12 更稳
            return -1.0
        improvement = (C_prev - C_t) / denom
        improvement = max(min(improvement, 1.0), -1.0) # 归一化到[-1, 1]
        return float(improvement)

    def _observation(self):
        """返回 dict obs：state=定长门序列(action id, padding=-1)，附带 action_mask。"""
        state = np.full(self.max_gates, -1, dtype=np.int32)
        if self.gate_tokens:
            L = min(len(self.gate_tokens), self.max_gates)
            state[:L] = np.array(self.gate_tokens[:L], dtype=np.int32)
        mask = self._action_mask()
        return {"state": state, "action_mask": mask}

    def _action_mask(self):
        """
        生成动作可行性掩码：1=legal, 0=illegal。
        基于 frontier（每个 qubit 的最外层触碰门）而不是全局 last_action。

        规则（门集: H,X,Y,Z,T,CNOT）：
        - max_gates 达到则全 0。
        - 无历史门则全 1。
        - cancellation：
            * 单比特自逆门 {H,X,Y,Z}：同门同 target -> illegal
            * CNOT：同向同 control/target -> illegal
        - redundancy（只做 Pauli-Pauli）：
            * X/Y/Z 两两组合等价于第三个 Pauli（允许全局相位）-> cand illegal
        - 不处理 1q <-> CNOT 的推门/传播冗余：若 frontier 是 CNOT，则对 1q cand 不做上述 1q 规则（视为屏障）。
        """
        action_n = len(self.actions)

        # # depth上限
        # if self.current_depth >= self.max_depth:
        #     return np.zeros(action_n, dtype=np.int8)

        # gates上限
        if self.current_gates >= self.task_budget:
            return np.zeros(action_n, dtype=np.int8)

        # 初始化为全 1，后续逐步屏蔽
        mask = np.ones(action_n, dtype=np.int8)

        # 配置
        cancel_1q = {'H', 'X', 'Y', 'Z'}

        pauli_redundant_pairs = {
            ('X', 'Y'), ('Y', 'X'),
            ('Y', 'Z'), ('Z', 'Y'),
            ('Z', 'X'), ('X', 'Z'),
        }
        # 仅允许对“激活”的量子比特施加操作：active_n_qubits 从任务里得知
        active_q = int(getattr(self, "active_n_qubits", self.n_qubits))

        def _frontier_act(q):
            """返回 qubit q 的最外层触碰门 act(dict)，若为空返回 None"""
            st = self._qubit_stacks[q]
            if not st:
                return None
            return self.actions[int(st[-1])]  # st[-1] 必须是 action_id(int)

        for idx, cand in enumerate(self.actions):
            g = cand['gate']

            # -------- 阻断未激活量子比特 --------
            if g in self.single_gates:
                if cand['target'] >= active_q:
                    mask[idx] = 0
                    continue
            elif g == 'CNOT':
                c = cand['control']
                t = cand['target']
                if c >= active_q or t >= active_q:
                    mask[idx] = 0
                    continue

            # ========= A) 单比特候选 =========
            if g in self.single_gates:
                q = cand['target']
                prev = _frontier_act(q)
                if prev is None:
                    continue  # 该 qubit 还没被触碰过 -> 不触发规则

                # 若 frontier 是 CNOT：按你的设定先不处理 1q<->CNOT 冗余/消除，直接放过
                if prev['gate'] == 'CNOT':
                    continue

                prev_gate = prev['gate']

                # 1) cancellation：H/X/Y/Z 同门同 target
                if g in cancel_1q and prev_gate == g:
                    mask[idx] = 0
                    continue

                # 2) redundancy：只处理 Pauli-Pauli (X/Y/Z)
                # （不看对易，只要两步能落回门集单门就 mask）
                if (prev_gate, g) in pauli_redundant_pairs:
                    mask[idx] = 0
                    continue

                continue

            # ========= B) CNOT 候选 =========
            if g == 'CNOT':
                c = cand['control']
                t = cand['target']

                prev_c = _frontier_act(c)
                prev_t = _frontier_act(t)

                # cancellation：只有当 control 和 target 两条线的 frontier 都是“同一个 CNOT(action)”
                # 且方向一致时，cand 才会与其相邻抵消
                if (prev_c is not None and prev_t is not None and
                    prev_c is prev_t and
                    prev_c['gate'] == 'CNOT' and
                    prev_c['control'] == c and prev_c['target'] == t):
                    mask[idx] = 0
                    continue

                continue

            # 其他类型动作（若未来扩展）：默认合法
            continue

        # 空电路直接返回当前屏蔽结果（需考虑未激活量子比特）
        if not self.gate_tokens:
            return mask

        return mask

    def reset(self, seed=None, options=None):
        """
        重置环境：从指定 mode 的任务列表随机抽取任务。
        重新设置量子比特数、动作映射、空间，以及缓存目标酉矩阵/目标态。
        同时清空 gate_tokens / qubit_stacks / fidelities 等 episode 状态，避免 mask 被污染。
        """
        super().reset(seed=seed)

        # 你采样任务用的是 self._np_random（不是 gym 的 self.np_random），所以这里也要跟着 reseed
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        # 采样任务
        self.current_task_id, self.target_qc, self.current_difficulty_bin, self.gates_target, self.active_n_qubits = self._sample_task()
        # tianshou 的 buffer.hasnull 会把 None 视为缺失值并直接报错（MalformedBufferError）
        # 有些任务缺少 difficulty_bin，统一用字符串占位，避免 None 进入 info
        safe_diff_bin = self.current_difficulty_bin if self.current_difficulty_bin is not None else "NA"
        # max_steps ≈ 1.3 * L，向上取整避免过早截断
        self.task_budget = max(1, int(np.ceil(self.gates_target * 1.3)))
        # ---- Route A: enforce fixed qubits ----
        task_nq = int(self.target_qc.num_qubits)
        env_nq = int(self.max_qubits)  # 也就是 init 传进来的 max_qubits
        if task_nq != env_nq:
            raise ValueError(
                f"[QUBITS MISMATCH] task has {task_nq} qubits, but env.max_qubits={env_nq}. "
                f"Fix by generating tasks with the same qubit count or by creating a separate env per qubit count."
            )
        # --------------------------------------


        # 重新构建当前电路与缓存
        self._qc = QuantumCircuit(self.n_qubits)
        self.qc_operator = Operator(self._qc)          # ✅ 与 _compute_fidelity() 一致
        self.current_depth = self._qc.depth()
        self.current_gates = len(self._qc.data)

        # 缓存目标（与你的 fidelity 计算一致）
        self.target_unitary = Operator(self.target_qc).data   # ✅ 与 _compute_fidelity() 一致

        # episode 状态清空（mask 依赖这些）
        self.gate_tokens = []
        self.fidelities = []
        self._qubit_stacks = [[] for _ in range(self.n_qubits)]  # ✅ frontier 从空开始


        obs = self._observation()
        info = {
            "fidelity": float(self._compute_fidelity()),
            "gate_count": self.current_gates,
            "step_count": len(self.gate_tokens),
            "depth": self.current_depth,
            "task_id": self.current_task_id,
            "n_qubits": int(self.n_qubits),
            "difficulty_bin": safe_diff_bin,
            "action_mask": obs["action_mask"],
            "valid_action_ratio": float(np.mean(obs["action_mask"])) if obs["action_mask"].size > 0 else 0.0,
        }
        return obs, info

    def step(self, action_id):
        t0 = time.perf_counter()
        # prev_fidelity
        if len(self.fidelities) == 0:
            prev_fidelity = float(self._compute_fidelity())
            self.fidelities = [prev_fidelity]
        else:
            prev_fidelity = float(self.fidelities[-1])

        # mask for current state
        mask = self._action_mask()
        mask_ratio = float(np.mean(mask)) if mask.size > 0 else 0.0
        self.profile["mask_valid_sum"] += mask_ratio
        self.profile["mask_calls"] += 1

        aid = -1 if action_id is None else int(action_id)
        illegal = False
        illegal_reason = None

        if action_id is None:
            illegal = True
            illegal_reason = "action is None"
        elif aid < 0 or aid >= len(self.actions):
            illegal = True
            illegal_reason = "action id out of range"
        elif mask[aid] == 0:
            illegal = True
            illegal_reason = "action is masked out (illegal)"

        operation = None if illegal else self._operation(aid)
        if not illegal and operation is None:
            illegal = True
            illegal_reason = "operation lookup returned None"

        if illegal:
            self.profile["illegal_count"] += 1
            obs = self._observation()
            mask_next = obs["action_mask"]
            safe_diff_bin = self.current_difficulty_bin if self.current_difficulty_bin is not None else "NA"
            info = {
                "fidelity": float(self.fidelities[-1]) if self.fidelities else 0.0,
                "step_count": len(self.gate_tokens),
                "gate_count": int(self.current_gates),
                "task_id": self.current_task_id,
                "n_qubits": self.n_qubits,
                "difficulty_bin": safe_diff_bin,
                "depth": int(self.current_depth),
                "action_mask": mask_next,
                "reached_max_depth": False,
                "reached_budget": False,
                "illegal_action": True,
                "illegal": True,
                "illegal_reason": illegal_reason,
                "valid_action_ratio": float(np.mean(mask_next)) if mask_next.size > 0 else 0.0,
            }
            self.profile["env_step_time"] += time.perf_counter() - t0
            self.profile["env_step_calls"] += 1
            # 非法动作：给予 fail_reward，截断本 episode 避免无限循环
            return obs, float(self.fail_reward), False, True, info

        self._qc.append(*operation)
        self.gate_tokens.append(aid)

        act = self.actions[aid]
        g = act['gate']
        if g in self.single_gates:
            q = act['target']
            self._qubit_stacks[q].append(aid)
        elif g == 'CNOT':
            c, t = act['control'], act['target']
            self._qubit_stacks[c].append(aid)
            self._qubit_stacks[t].append(aid)

        self.qc_operator = Operator(self._qc)
        self.current_depth = self._qc.depth()
        self.current_gates = len(self._qc.data)

        fidelity = float(self._compute_fidelity())

        reached_fidelity = fidelity >= float(self.fidelity_threshold)
        reached_max_depth = int(self.current_depth) >= int(self.max_depth)
        reached_budget = int(self.current_gates) >= int(self.task_budget)  # ✅ 若你保留 max_gates

        terminated = bool(reached_fidelity)
        # truncated = bool(reached_max_depth)          # ✅ 避免死循环
        truncated = bool((not terminated) and reached_budget)

        reward = float(self.reward_cost(
            fidelity=fidelity,
            prev_fidelity=prev_fidelity,
            current_depth=int(self.current_depth),
            current_gates=int(self.current_gates)
        ))

        obs = self._observation()
        mask_next = obs["action_mask"]
        safe_diff_bin = self.current_difficulty_bin if self.current_difficulty_bin is not None else "NA"
        info = {
            "fidelity": fidelity,
            "step_count": len(self.gate_tokens),
            "gate_count": int(self.current_gates),
            "task_id": self.current_task_id,
            "n_qubits": self.n_qubits,
            "difficulty_bin": safe_diff_bin,
            "depth": int(self.current_depth),
            "action_mask": mask_next,
            "reached_max_depth": bool(reached_max_depth),
            "reached_budget": bool(reached_budget),
            "illegal_action": False,
            "illegal": False,
            "valid_action_ratio": float(np.mean(mask_next)) if mask_next.size > 0 else 0.0,
        }

        self.fidelities.append(fidelity)
        
        # 调试模式：检查 gate_tokens 与 qc.data 长度一致
        if getattr(self, "debug", False):
            assert len(self.gate_tokens) == len(self._qc.data), "gate_tokens 与 qc.data 长度不一致"

        self.profile["env_step_time"] += time.perf_counter() - t0
        self.profile["env_step_calls"] += 1

        return obs, reward, terminated, truncated, info

    def get_circuit_stats(self):
        """返回当前电路的统计信息，用于日志/排名。"""
        gate_count = len(self._qc.data)
        depth = int(self._qc.depth())
        cx_count = sum(1 for inst, _, _ in self._qc.data if isinstance(inst, CXGate))
        oneq_count = gate_count - cx_count
        return {
            "gate_count": int(gate_count),
            "depth": depth,
            "cx_count": int(cx_count),
            "oneq_count": int(oneq_count),
        }

    def render(self):
        """渲染当前电路"""
        if self.render_mode is None:
            return None
        return self._qc.draw(self.render_mode)

    def get_action_info(self, action_id):
        """调试：返回动作对应的门及作用量子比特"""
        if 0 <= int(action_id) < len(self.actions):
            return self.actions[int(action_id)]
        return None

# 注册环境
register(
    id='CircuitDesigner-Discrete-v1',
    entry_point='env:CircuitDesignerDiscrete',
)
