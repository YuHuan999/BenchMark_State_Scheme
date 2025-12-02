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
GATES = 6  # H, X, Y, Z, T, CNOT

class CircuitDesigner(gym.Env):
    """ 量子电路环境：逐门构建量子电路以达成指定目标 """

    metadata = {"render_modes": ["image","text"], "render_fps": 30}

    def __init__(self, max_qubits: int, max_depth: int, objective: str, 
                 punish=False, seed=None, render_mode=None, fidelity_threshold=0.99):
        super().__init__()
        if seed is not None: 
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        
        self.render_mode = render_mode
        self.name = f"{objective}|{max_qubits}-{max_depth}"

        # 环境参数
        self.qubits, self.depth = max_qubits, max_depth
        self.max_steps = self.depth
        self.punish = punish
        self.objective = objective
        self.target = self._target(*objective.split('-'), seed)
        self._qc = QuantumCircuit(self.qubits)
        self.qc_unitary = Operator(self._qc)
        self.last_fidelity = 0
        self.last_cost = 0
        self.fidelity_threshold = fidelity_threshold
        self.reached = False
        
        # 观察空间和动作空间
        dummy_obs, _ = self._state_current_state()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=dummy_obs.shape, dtype=np.float32
        )

        # 动作空间: [门类型, 目标量子比特, 控制量子比特]
        m = 1e-5  # 防止边界溢出
        self.action_space = gym.spaces.Box(
            np.array([0, 0, 0]), 
            np.array([GATES-m, self.qubits-m, self.qubits-m]),
            dtype=np.float32
        )

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
                
        if task == 'UC':  # 酉矩阵合成
            if target == 'random': 
                return random_unitary(2**self.qubits, seed)
            _t = QuantumCircuit(self.qubits)
            if target == 'hadamard': 
                _t.append(HGate(), [0])
            if target == 'toffoli': 
                assert self.qubits >= 3, "Toffoli门需要至少3个量子比特"
                _t.append(CCXGate(), [0,1,2])
            return Operator(_t)
            
        raise ValueError(f'未定义的目标: {task}-{target}')

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

    def _operation(self, action):
        """将动作转换为量子门操作"""
        gate, wire, cntrl = action
        gate, wire, cntrl = np.floor([gate, wire, cntrl]).astype(int)
        
        # 确保索引有效
        wire = max(0, min(wire, self.qubits-1))
        cntrl = max(0, min(cntrl, self.qubits-1))
        
        # 单量子比特门
        if gate == 0: return HGate(), [wire]
        if gate == 1: return XGate(), [wire]
        if gate == 2: return YGate(), [wire]
        if gate == 3: return ZGate(), [wire]
        if gate == 4: return TGate(), [wire]
        
        # 双量子比特门
        if gate == 5: 
            return CXGate(), [cntrl, wire]
        
        # 默认返回恒等操作
        return None

    def value_delta(self, fidelity, cost):
        """计算奖励和成本的变化"""
        reward = fidelity - self.last_fidelity
        cost_delta = cost - self.last_cost
        self.last_fidelity = fidelity
        self.last_cost = cost
        return reward, cost_delta

    def _reward(self):
        """计算奖励"""
        if 'SP' in self.objective: 
            # 状态制备任务
            current_state = Statevector.from_instruction(self._qc).data
            target_state = self.target.data
            fidelity = abs(np.vdot(current_state, target_state))**2
            
        elif 'UC' in self.objective:
            # 酉矩阵编译任务
            U_current = self.qc_unitary.data
            U_target = self.target.data
            hs_inner = np.trace(U_current.conj().T @ U_target) / U_target.shape[0]
            fidelity = np.abs(hs_inner) ** 2
        
        # 检查是否达到目标
        if fidelity >= self.fidelity_threshold:
            self.reached = True
            
        # 计算成本（深度惩罚）
        cost = max(0, self._qc.depth() - self.depth/3) / (self.depth / 2 * 3)
        
        # 计算奖励变化
        reward, cost_delta = self.value_delta(fidelity, cost)
        
        # 额外奖励和惩罚
        if self.reached:
            reward += 1.0  # 达到目标的奖励
        
        reward -= 0.01  # 小的步骤惩罚，鼓励更短的电路
        
        return reward, cost_delta

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.last_cost = 0
        self.last_fidelity = 0
        self._qc.clear()
        self.reached = False
        self.qc_unitary = Operator(self._qc)
        
        observation, info = self._state_current_state()
        return observation, info

    def step(self, action):
        """执行一步动作"""
        operation = self._operation(action)
        
        if operation is not None: 
            self._qc.append(*operation) 
            self.qc_unitary = Operator(self._qc)
        
        # 计算奖励
        reward, cost = self._reward()
        
        # 获取状态
        state, info = self._state_current_state()
        
        # 检查终止条件
        terminated = self.reached
        truncated = (self._qc.depth() >= self.depth or 
                    self._operations >= self.max_steps)
        
        if terminated:
            info['termination_reason'] = 'TARGET_REACHED'
        elif truncated:
            info['termination_reason'] = 'MAX_LIMIT_REACHED'
        
        # 更新信息
        info.update({
            'fidelity': self.last_fidelity,
            'cost': cost,
            'reward': reward
        })
        
        if self.punish: 
            reward -= cost
        


        
        return state, reward, terminated, truncated, info

    def render(self):
        """渲染电路"""
        if self.render_mode is None: 
            return None
        return self._qc.draw(self.render_mode)


# 注册环境
register(
    id='CircuitDesigner-v1',
    entry_point='envs.circuit_sys:CircuitDesigner',
)
