"""
LightZero 环境包装器
将 CircuitDesignerDiscrete 适配为 LightZero 的 BaseEnv 接口
"""

import numpy as np
from easydict import EasyDict
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from gymnasium import spaces
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from .circuit_sys_discrete import CircuitDesignerDiscrete


@ENV_REGISTRY.register('circuit_designer_lightzero')
class CircuitDesignerLightZeroEnv(BaseEnv):
    """
    LightZero 环境包装器，用于量子电路设计环境
    """
    
    def __init__(self, cfg: EasyDict = None):
        """
        初始化环境
        
        Args:
            cfg: 环境配置字典，包含:
                - max_qubits: 最大量子比特数
                - max_gates: 最大门数量
                - objective: 目标任务（如'SP-bell'）
                - punish: 是否使用惩罚项
                - fidelity_threshold: 保真度阈值
                - save_dir: episode数据保存目录（可选）
                - enable_monitor: 是否启用episode监控（默认True）
                - env_id: 环境ID（用于多进程时区分不同环境）
        """
        if cfg is None:
            cfg = EasyDict({
                'max_qubits': 2,
                'max_gates': 5,
                'objective': 'SP-bell',
                'punish': False,
                'fidelity_threshold': 0.99,
                'enable_monitor': True
            })
        
        # 创建原始环境
        self.env = CircuitDesignerDiscrete(
            max_qubits=cfg.max_qubits,
            max_gates=cfg.max_gates,
            objective=cfg.objective,
            punish=cfg.punish,
            fidelity_threshold=cfg.fidelity_threshold
        )
        
        # 保存配置
        self.cfg = cfg
        
        # Episode记录相关
        self.enable_monitor = cfg.get('enable_monitor', True)
        self.save_dir = cfg.get('save_dir', None)
        # 使用进程ID+时间戳生成唯一的env_id
        import time
        self.env_id = cfg.get('env_id', f"{os.getpid()}_{int(time.time()*1000000) % 1000000}")
        
        # Episode数据缓存
        self._episode_data = []
        self._episode_count = 0
        self._current_episode_rewards = []
        self._current_episode_actions = []
        
        # 初始化CSV文件
        if self.enable_monitor and self.save_dir:
            self._init_episode_files()
        
        # AlphaZero 兼容属性
        self.battle_mode = 'self_play_mode'
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.current_player = 0  # 单玩家游戏
        
        # 当前状态缓存
        self._current_obs = None
        
        # 累积奖励（用于eval_episode_return）
        self._episode_reward = 0
        
        # 随机种子
        self._seed = None
        self._dynamic_seed = True
        
    @property
    def observation_space(self):
        """观察空间"""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """动作空间"""
        return self.env.action_space
    
    @property
    def reward_space(self):
        """奖励空间"""
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
    
    @property
    def legal_actions(self):
        """
        返回当前合法的动作列表
        对于量子电路环境，所有动作都是合法的
        """
        return list(range(self.action_space.n))
    
    def current_state(self):
        """
        返回当前状态（用于MCTS模拟）
        
        Returns:
            tuple: (原始状态, 归一化状态)
        """
        if self._current_obs is None:
            obs, _ = self.env._state_current_state()
            self._current_obs = obs
        return (self._current_obs, self._current_obs)
    
    def get_done_winner(self):
        """
        检查游戏是否结束及获胜者
        对于单玩家优化任务，"获胜"意味着达到目标保真度
        
        Returns:
            tuple: (是否结束, 获胜者列表)
                - 成功达到目标: (True, [0])
                - 达到最大限制但未成功: (True, [-1])
                - 未完成: (False, [-1])
        """
        # 检查是否达到目标
        if hasattr(self.env, 'reached') and self.env.reached:
            return True, [0]  # 达到目标，玩家0获胜
        
        # 检查是否超过最大门数（失败）
        if len(self.env._qc.data) >= self.env.max_gates:
            return True, [-1]  # 达到最大门数但未成功，游戏结束但失败
        
        return False, [-1]  # 游戏未完成
    
    def reset(self, start_player_index=0, init_state=None, **kwargs):
        """
        重置环境
        
        Args:
            start_player_index: 起始玩家索引（单玩家游戏忽略）
            init_state: 初始状态（可选）
            **kwargs: 其他参数
            
        Returns:
            dict: 观察字典，包含:
                - observation: 原始观察
                - board: AlphaZero期望的棋盘状态
                - current_player_index: 当前玩家索引
                - action_mask: 动作掩码（所有动作可用）
                - to_play: 轮到谁下（单玩家固定为0）
        """
        obs, info = self.env.reset()
        self._current_obs = obs
        self._episode_reward = 0  # 重置累积奖励
        self._current_episode_rewards = []  # 重置episode奖励列表
        self._current_episode_actions = []  # 重置episode动作列表
        
        # 创建动作掩码（所有动作都可用）
        action_mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # 返回 AlphaZero 期望的格式（只返回obs字典，不返回info）
        obs_dict = {
            'observation': obs,
            'board': obs,  # AlphaZero期望的字段
            'current_player_index': 0,  # 单玩家游戏
            'action_mask': action_mask,
            'to_play': 0  # 单玩家游戏固定为0
        }
        
        return obs_dict
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 动作ID
            
        Returns:
            BaseEnvTimestep: 包含obs, reward, done, info的时间步
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = obs
        
        # 记录当前步的数据
        self._current_episode_rewards.append(reward)
        self._current_episode_actions.append(action)
        
        # 创建动作掩码
        action_mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # 创建观察字典
        obs_dict = {
            'observation': obs,
            'board': obs,
            'current_player_index': 0,
            'action_mask': action_mask,
            'to_play': 0  # 单玩家游戏固定为0
        }
        
        # 累积奖励
        self._episode_reward += reward
        
        # LightZero 使用 terminated 表示游戏结束
        done = terminated or truncated
        
        # Episode结束时记录数据
        if done:
            info['eval_episode_return'] = self._episode_reward
            
            # 从info中提取真实数据
            final_fidelity = info.get('fidelity', 0.0)
            episode_length = len(self._current_episode_actions)
            success = final_fidelity >= self.cfg.fidelity_threshold
            
            # 记录episode数据（使用环境返回的真实值）
            # 注意：total_reward是归一化后的值，与PPO/A2C/DQN保持一致
            self._record_episode(
                total_reward=self._episode_reward,  # 使用归一化后的累积reward
                length=episode_length,
                final_fidelity=final_fidelity,
                success=success
            )
            
            # 清空当前episode的缓存
            self._current_episode_rewards = []
            self._current_episode_actions = []
        
        # 返回BaseEnvTimestep
        return BaseEnvTimestep(obs_dict, reward, done, info)
    
    def _init_episode_files(self):
        """初始化episode记录文件"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径（每个环境有独立的临时文件）
        self.csv_path = os.path.join(self.save_dir, f'episodes_env{self.env_id}.csv')
        
        # 如果文件不存在，创建并写入表头
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'episode', 'total_reward', 'length', 'final_fidelity', 'success', 'timestamp'
            ])
            df.to_csv(self.csv_path, index=False)
    
    def _record_episode(self, total_reward, length, final_fidelity, success):
        """记录一个episode的数据"""
        if not self.enable_monitor or not self.save_dir:
            return
        
        self._episode_count += 1
        
        episode_data = {
            'episode': self._episode_count,
            'total_reward': float(total_reward),
            'length': int(length),
            'final_fidelity': float(final_fidelity),
            'success': bool(success),
            'timestamp': datetime.now().isoformat()
        }
        
        self._episode_data.append(episode_data)
        
        # 每100个episodes保存一次
        if len(self._episode_data) >= 100:
            self._save_episodes()
    
    def _save_episodes(self):
        """保存累积的episode数据到CSV"""
        if len(self._episode_data) == 0:
            return
        
        df = pd.DataFrame(self._episode_data)
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        self._episode_data = []
    
    def close(self):
        """关闭环境"""
        # 保存剩余的episode数据
        if self.enable_monitor and self.save_dir:
            self._save_episodes()
        
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def seed(self, seed: int, dynamic_seed: bool = True):
        """设置随机种子"""
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        return [seed]
    
    def random_action(self):
        """返回一个随机动作"""
        return self.action_space.sample()
    
    def __repr__(self):
        return f"<CircuitDesignerLightZeroEnv({self.env.name})>"
