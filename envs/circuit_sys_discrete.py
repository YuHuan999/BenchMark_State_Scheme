import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, TGate, CXGate, CCXGate
from qiskit.quantum_info import Statevector, Operator, random_statevector, random_unitary
from qiskit.converters import circuit_to_dag

import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)

flat = lambda s: np.concatenate([s.data.real, s.data.imag]).astype(np.float32).flatten()

class FidelityLengthReward:
    """
    终点惩罚型奖励函数（episodic penalty）
    当电路达到目标保真度或最大门数时结束。
    奖励 = 最终保真度 - λ * (门数量 / 最大门数)
    """
    def __init__(self, fidelity_threshold=0.99, penalty_coeff=0.02, max_gates=10):
        self.fidelity_threshold = fidelity_threshold
        self.penalty_coeff = penalty_coeff
        self.max_gates = max_gates

    def compute(self, fidelity, gate_count):
        reached_fidelity = fidelity >= self.fidelity_threshold
        reached_max_gates = gate_count >= self.max_gates
        done = reached_fidelity or reached_max_gates

        # 只在终点给奖励
        if done:
            reward = fidelity - self.penalty_coeff * (gate_count / self.max_gates)
            if reached_fidelity:
                reward += 1.0  # 成功额外奖励
            
            # 将奖励归一化到[-1, 1]范围，以匹配value head的输出范围
            # 原始范围: [-0.02, 2.0] → 归一化到 [-1, 1]
            reward_min, reward_max = -0.02, 2.0
            reward_normalized = 2.0 * (reward - reward_min) / (reward_max - reward_min) - 1.0
        else:
            reward = 0.0  # 中间步骤不给奖励
            reward_normalized = 0.0  # 中间步骤归一化后也是0

        info = dict(
            fidelity=fidelity,
            gate_count=gate_count,
            reached_fidelity=reached_fidelity,
            reached_max_gates=reached_max_gates,
            reward_original=reward,  # 保存原始奖励用于调试
        )
        return reward_normalized, done, info

class CircuitDesignerDiscrete(gym.Env):
    """ 量子电路环境：适合DQN的离散动作空间版本 """

    metadata = {"render_modes": ["image","text"], "render_fps": 30}

    def __init__(self, max_qubits: int, max_gates: int, objective: str, 
                 punish=False, seed=None, render_mode=None, fidelity_threshold=0.99):
        super().__init__()
        if seed is not None: 
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        
        self.render_mode = render_mode
        self.name = f"{objective}|{max_qubits}-g{max_gates}"

        # 环境参数
        self.qubits = max_qubits
        self.max_gates = max_gates
        self.max_steps = max_gates  # 最大步数等于最大门数
        self.punish = punish
        self.objective = objective
        self.target = self._target(*objective.split('-'), seed)
        self._qc = QuantumCircuit(self.qubits)
        self.qc_unitary = Operator(self._qc)
        self.last_fidelity = 0
        self.last_cost = 0
        self.fidelity_threshold = fidelity_threshold
        self.reached = False
        
        # 初始化奖励函数
        self.reward_fn = FidelityLengthReward(
            fidelity_threshold=fidelity_threshold,
            penalty_coeff=0.02,
            max_gates=max_gates
        )
        
        # 定义门类型
        self.gates = ['H', 'X', 'Y', 'Z', 'T', 'CNOT']
        self.single_gates = ['H', 'X', 'Y', 'Z', 'T']  # 单量子比特门
        self.two_gates = ['CNOT']  # 双量子比特门
        
        # 观察空间S
        dummy_obs, _ = self._state_current_state()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=dummy_obs.shape, dtype=np.float32
        )

        # 离散动作空间：为每种可能的门-量子比特组合分配一个动作ID
        self.action_mapping = self._create_action_mapping()
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))
        
        print(f"Created {len(self.action_mapping)} discrete actions")

    def _create_action_mapping(self):
        """创建动作ID到(门类型, 量子比特)的映射"""
        actions = []
        
        # 单量子比特门：每个门在每个量子比特上
        for gate in self.single_gates:
            for qubit in range(self.qubits):
                actions.append({
                    'gate': gate,
                    'target': qubit,
                    'control': None
                })
        
        # 双量子比特门：每个门在每对不同量子比特上
        for gate in self.two_gates:
            for control in range(self.qubits):
                for target in range(self.qubits):
                    if control != target:  # 控制和目标量子比特必须不同
                        actions.append({
                            'gate': gate,
                            'target': target,
                            'control': control
                        })
        
        return actions

    def _target(self, task, target, seed):
        """创建目标状态或酉矩阵"""
        if task == 'SP':  # 状态制备
            if target == 'random': 
                return random_statevector(2**self.qubits, seed)
            if target == 'bell': 
                return Statevector(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128))
            if 'ghz' in target: 
                n = int(target[-1])
                assert 2 <= n <= self.qubits, f"GHZ需要2到{self.qubits}个量子比特"
                target_vec = np.zeros(shape=(2**n,), dtype=np.complex128)
                target_vec[0] = target_vec[-1] = 1/np.sqrt(2)
                return Statevector(target_vec) 
        
        if task == 'UC':  # 酉矩阵编译
            if target == 'random': 
                return random_unitary(2**self.qubits, seed)
            _t = QuantumCircuit(self.qubits)
            if target == 'hadamard': 
                _t.append(HGate(), [0])
            if target == 'toffoli': 
                assert self.qubits >= 3, "Toffoli门需要至少3个量子比特"
                _t.append(CCXGate(), [0,1,2])
            return Operator(_t)
        
        raise ValueError(f'未定义的任务: {task}-{target}')

    @property
    def _operations(self): 
        return sum([v for k,v in self._qc.count_ops().items()])

    @property 
    def _used_wires(self): 
        return self.qubits - len(list(circuit_to_dag(self._qc).idle_wires()))

    def _state_current_state(self):
        """计算当前状态观察"""
        info = {
            'depth': self._qc.depth(), 
            'operations': self._operations, 
            'used_wires': self._used_wires
        }
        
        V_t = self.qc_unitary  # 当前电路的酉矩阵
        
        if 'UC' in self.objective: 
            # 酉矩阵编译：计算 V_t† @ U
            U = self.target  # 目标酉矩阵
            state = V_t.data.conj().T @ U.data
            state_flat = state.flatten()
            observation = np.concatenate([state_flat.real, state_flat.imag]).astype(np.float32)
            return observation, info
            
        if 'SP' in self.objective:  
            # 状态制备：计算 V_t† |target⟩
            target_state = self.target
            result_state = target_state.evolve(V_t.adjoint())
            observation = flat(result_state)
            return observation, info

    def _operation(self, action_id):
        """将动作ID转换为量子门操作"""
        if action_id >= len(self.action_mapping):
            return None
            
        action = self.action_mapping[action_id]
        gate_type = action['gate']
        target = action['target']
        control = action['control']
        
        # 单量子比特门
        if gate_type == 'H': return HGate(), [target]
        if gate_type == 'X': return XGate(), [target]
        if gate_type == 'Y': return YGate(), [target]
        if gate_type == 'Z': return ZGate(), [target]
        if gate_type == 'T': return TGate(), [target]
        
        # 双量子比特门
        if gate_type == 'CNOT': 
            return CXGate(), [control, target]
        
        return None

    def value_delta(self, fidelity, cost):
        """计算奖励和成本的变化"""
        reward = fidelity - self.last_fidelity
        cost_delta = cost - self.last_cost
        self.last_fidelity = fidelity
        self.last_cost = cost
        return reward, cost_delta

    def _reward(self):
        """计算当前 fidelity，但不再直接生成 reward"""
        if 'SP' in self.objective:
            current_state = Statevector.from_instruction(self._qc).data
            target_state = self.target.data
            fidelity = abs(np.vdot(current_state, target_state)) ** 2
        elif 'UC' in self.objective:
            U_current = self.qc_unitary.data
            U_target = self.target.data
            hs_inner = np.trace(U_current.conj().T @ U_target) / U_target.shape[0]
            fidelity = np.abs(hs_inner) ** 2
        else:
            fidelity = 0.0
        return fidelity

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.last_fidelity = 0
        self.last_cost = 0
        self.reached = False
        self._qc.clear()
        self.qc_unitary = Operator(self._qc)
        
        observation, info = self._state_current_state()
        return observation, info

    def step(self, action_id):
        """执行一步动作（使用新的终点惩罚奖励函数）"""
        operation = self._operation(action_id)
        if operation is not None:
            self._qc.append(*operation)
            self.qc_unitary = Operator(self._qc)

        # 当前 fidelity 和门数量
        fidelity = self._reward()
        gate_count = len(self._qc.data)  # 使用门数量，不再使用depth()

        # 计算奖励与是否终止
        reward, done, reward_info = self.reward_fn.compute(fidelity, gate_count)

        # 判断终止原因
        if reward_info['reached_fidelity']:
            termination_reason = 'FIDELITY_REACHED'
        elif reward_info['reached_max_gates']:
            termination_reason = 'MAX_GATES_REACHED'
        else:
            termination_reason = None

        # 获取状态
        obs, info = self._state_current_state()

        # 构造 info 字典
        info.update({
            'fidelity': fidelity,
            'reward': reward,
            'reward_original': reward_info.get('reward_original', reward),  # 添加原始reward
            'termination_reason': termination_reason,
            'gate_count': gate_count,
            'circuit_depth': self._qc.depth()  # 保留depth信息用于调试
        })

        # punish 模式下附加代价
        if self.punish:
            reward -= 0.01 * (gate_count / self.max_gates)

        terminated = done and reward_info['reached_fidelity']
        truncated = done and reward_info['reached_max_gates']

        return obs, reward, terminated, truncated, info

        
    def render(self):
        """渲染电路"""
        if self.render_mode is None: 
            return None
        return self._qc.draw(self.render_mode)

    def get_action_info(self, action_id):
        """获取动作的详细信息（用于调试）"""
        if action_id < len(self.action_mapping):
            return self.action_mapping[action_id]
        return None


# 注册环境
register(
    id='CircuitDesigner-Discrete-v1',
    entry_point='envs.circuit_sys_discrete:CircuitDesignerDiscrete',
)
