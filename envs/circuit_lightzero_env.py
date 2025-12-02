"""
LightZero 环境包装器
将 CircuitDesignerDiscrete 适配为 LightZero 的 BaseEnv 接口
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from easydict import EasyDict
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from gymnasium import spaces

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
        
        # 🔥 启用环境内置记录（更可靠！）
        self.enable_monitor = cfg.get('enable_monitor', True)
        self.save_dir = cfg.get('save_dir', None)
        
        # 🔥 如果没有save_dir，但启用了monitor，自动创建一个默认目录
        if self.enable_monitor and self.save_dir is None:
            # 从objective生成默认目录名
            task_name = cfg.objective.replace('-', '_')
            self.save_dir = f'results/AlphaZero_{task_name}_q{cfg.max_qubits}_g{cfg.max_gates}/episodes'
        
        # 使用进程ID+时间戳生成唯一的env_id
        self.env_id = cfg.get('env_id', f"{os.getpid()}_{int(time.time()*1000000) % 1000000}")
        
        # Episode数据缓存
        self._episode_data = []
        self._episode_count = 0
        self._current_episode_rewards = []
        self._current_episode_actions = []
        
        # 轨迹记录
        self._current_trajectory = {
            'step_rewards': [],
            'step_fidelities': [],
            'step_actions': [],
            'step_gate_counts': [],
            'step_circuit_depths': []
        }
        
        # 🔥 训练进度追踪（用于training_progress.csv）
        self._total_timesteps = 0  # 总环境交互步数
        self._progress_window = []  # 最近100个episodes的统计
        self._last_progress_timesteps = 0  # 上次保存进度的timesteps
        self._progress_interval = 10000  # 🔥 每200步保存一次进度（便于测试）
        
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
        
        # 🔥 重置轨迹记录
        self._current_trajectory = {
            'step_rewards': [],
            'step_fidelities': [],
            'step_actions': [],
            'step_gate_counts': [],
            'step_circuit_depths': []
        }
        
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
        
        # 🔥 追踪总timesteps
        self._total_timesteps += 1
        
        # 记录当前步的数据
        self._current_episode_rewards.append(reward)
        self._current_episode_actions.append(action)
        
        # 🔥 记录轨迹信息（每一步）
        self._current_trajectory['step_rewards'].append(float(reward))
        self._current_trajectory['step_fidelities'].append(float(info.get('fidelity', 0.0)))
        self._current_trajectory['step_actions'].append(int(action))
        self._current_trajectory['step_gate_counts'].append(int(info.get('gate_count', 0)))
        self._current_trajectory['step_circuit_depths'].append(int(info.get('circuit_depth', 0)))
        
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
            
            # 🔥 从环境的info直接提取真实数据
            final_fidelity = info.get('fidelity', 0.0)
            gate_count = info.get('gate_count', 0)
            episode_length = len(self._current_episode_actions)
            success = final_fidelity >= self.cfg.fidelity_threshold
            
            # 🔥 添加到进度窗口（用于training_progress.csv）
            self._progress_window.append({
                'total_reward': float(self._episode_reward),
                'fidelity': float(final_fidelity),
                'length': int(gate_count),
                'success': float(success)
            })
            
            # 保持窗口大小为100
            if len(self._progress_window) > 100:
                self._progress_window.pop(0)
            
            # 🔥 检查是否需要保存训练进度
            if self._total_timesteps - self._last_progress_timesteps >= self._progress_interval:
                self._save_training_progress()
                self._last_progress_timesteps = self._total_timesteps
            
            # 记录episode数据（包含轨迹）
            self._record_episode(
                total_reward=self._episode_reward,
                length=episode_length,
                final_fidelity=final_fidelity,
                gate_count=gate_count,
                success=success,
                trajectory=self._current_trajectory.copy()  # 传递轨迹副本
            )
            
            # 清空当前episode的缓存
            self._current_episode_rewards = []
            self._current_episode_actions = []
            # 重置轨迹
            self._current_trajectory = {
                'step_rewards': [],
                'step_fidelities': [],
                'step_actions': [],
                'step_gate_counts': [],
                'step_circuit_depths': []
            }
        
        # 返回BaseEnvTimestep
        return BaseEnvTimestep(obs_dict, reward, done, info)
    
    def _init_episode_files(self):
        """初始化episode记录文件"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径（每个环境有独立的临时文件）
        self.csv_path = os.path.join(self.save_dir, f'episodes_env{self.env_id}.csv')
        
        # 🔥 JSONL文件路径（详细轨迹）
        self.jsonl_path = os.path.join(self.save_dir, f'episodes_detailed_env{self.env_id}.jsonl')
        
        # 🔥 训练进度文件路径（所有环境共享，追加模式）
        self.progress_path = os.path.join(self.save_dir, 'training_progress.csv')
        
        # 🔥 使用与DQN相同的CSV格式
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'episode', 'total_reward', 'final_fidelity', 'length', 'success'
            ])
            df.to_csv(self.csv_path, index=False)
        
        # 初始化training_progress.csv（只初始化一次）
        if not os.path.exists(self.progress_path):
            df_progress = pd.DataFrame(columns=[
                'timesteps', 'mean_reward', 'std_reward', 'mean_fidelity', 
                'std_fidelity', 'success_rate', 'mean_length'
            ])
            df_progress.to_csv(self.progress_path, index=False)
    
    def _record_episode(self, total_reward, length, final_fidelity, gate_count, success, trajectory=None):
        """记录一个episode的数据"""
        if not self.enable_monitor or not self.save_dir:
            return
        
        self._episode_count += 1
        
        # 🔥 使用与DQN完全相同的格式
        # DQN格式：episode, total_reward, length(实际是gate_count), fidelity, success
        episode_data = {
            'episode': self._episode_count,
            'total_reward': float(total_reward),
            'final_fidelity': int(gate_count),  # 🔥 第3列实际是gate_count！
            'length': float(final_fidelity),    # 🔥 第4列实际是fidelity！
            'success': bool(success),
        }
        
        self._episode_data.append(episode_data)
        
        # 🔥 保存详细轨迹到JSONL
        if trajectory is not None:
            self._save_trajectory(
                episode=self._episode_count,
                total_reward=total_reward,
                length=length,
                final_fidelity=final_fidelity,
                success=success,
                trajectory=trajectory
            )
        
        # 每100个episodes保存一次
        if len(self._episode_data) >= 100:
            self._save_episodes()
            self._trigger_merge()  # 🔥 触发CSV合并
    
    def _save_episodes(self):
        """保存累积的episode数据到CSV"""
        if len(self._episode_data) == 0:
            return
        
        df = pd.DataFrame(self._episode_data)
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        self._episode_data = []
    
    def _save_trajectory(self, episode, total_reward, length, final_fidelity, success, trajectory):
        """保存详细轨迹到JSONL文件"""
        import json
        
        # 构造轨迹记录（与DQN格式一致，添加env_id标识）
        trajectory_record = {
            'env_id': self.env_id,  # 🔥 标识来自哪个环境
            'episode': episode,
            'total_reward': float(total_reward),
            'length': int(length),
            'final_fidelity': float(final_fidelity),
            'success': bool(success),
            'trajectory': trajectory
        }
        
        # 追加到JSONL文件
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(trajectory_record) + '\n')
    
    def _save_training_progress(self):
        """保存训练进度统计到training_progress.csv"""
        if len(self._progress_window) == 0:
            return
        
        import numpy as np
        
        # 计算统计量（基于滑动窗口）
        rewards = [ep['total_reward'] for ep in self._progress_window]
        fidelities = [ep['fidelity'] for ep in self._progress_window]
        lengths = [ep['length'] for ep in self._progress_window]
        successes = [ep['success'] for ep in self._progress_window]
        
        progress_record = {
            'timesteps': self._total_timesteps,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_fidelity': float(np.mean(fidelities)),
            'std_fidelity': float(np.std(fidelities)),
            'success_rate': float(np.mean(successes)),
            'mean_length': float(np.mean(lengths))
        }
        
        # 追加到CSV（使用锁避免多进程冲突）
        try:
            df = pd.DataFrame([progress_record])
            df.to_csv(self.progress_path, mode='a', header=False, index=False)
        except Exception as e:
            # 静默处理错误（可能是多进程写入冲突）
            pass
    
    def _trigger_merge(self):
        """触发CSV合并（异步，不阻塞环境运行）"""
        try:
            # 导入合并器
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from csv_merger import merge_csv_once
            
            # 执行合并
            merge_csv_once(self.save_dir)
        except Exception as e:
            # 打印错误便于调试（只在前几次）
            if hasattr(self, '_merge_error_count'):
                self._merge_error_count += 1
            else:
                self._merge_error_count = 1
            
            if self._merge_error_count <= 3:
                print(f"⚠️  CSV合并失败 (#{self._merge_error_count}): {e}")
    
    def close(self):
        """关闭环境"""
        # 🔥 保存剩余的episode数据
        if self.enable_monitor and self.save_dir:
            self._save_episodes()
            self._trigger_merge()  # 最后一次合并
        
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
