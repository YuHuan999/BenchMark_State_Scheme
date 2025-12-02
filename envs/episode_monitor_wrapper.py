"""
Episode Monitor Wrapper for AlphaZero
精确记录每个episode的详细数据，与DQN/PPO/A2C格式完全一致
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional


class EpisodeMonitorWrapper:
    """
    环境wrapper，用于记录每个episode的详细信息
    与Stable-Baselines3的Monitor功能相同
    """
    
    def __init__(self, env, save_dir: str, batch_size: int = 100):
        """
        Args:
            env: 要包装的环境
            save_dir: 保存目录
            batch_size: 每多少个episodes保存一次
        """
        self.env = env
        self.save_dir = save_dir
        self.batch_size = batch_size
        
        # 暴露环境的基本属性
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.name = getattr(env, 'name', 'quantum_circuit')
        self.seed = getattr(env, 'seed', None)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # Episode计数
        self.episode_count = 0
        self.total_episodes = 0
        
        # 当前episode的数据
        self.current_episode = {
            'step_rewards': [],
            'step_fidelities': [],
            'step_actions': [],
            'step_gate_counts': [],
        }
        
        # 批量缓存
        self.batch_cache = []
        
        # 文件路径
        self.csv_path = os.path.join(save_dir, 'episodes_episodes.csv')
        self.jsonl_path = os.path.join(save_dir, 'episodes_detailed.jsonl')
        self.stats_path = os.path.join(save_dir, 'episodes_statistics.json')
        
        # 初始化CSV文件
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'episode', 'total_reward', 'length', 'final_fidelity', 'success'
            ])
            df.to_csv(self.csv_path, index=False)
        else:
            # 如果文件存在，读取已有的episode数量
            try:
                df = pd.read_csv(self.csv_path)
                self.total_episodes = len(df)
                print(f"📊 继续记录，已有{self.total_episodes}个episodes")
            except:
                self.total_episodes = 0
    
    def reset(self, **kwargs):
        """重置环境并开始新的episode"""
        # 清空当前episode数据
        self.current_episode = {
            'step_rewards': [],
            'step_fidelities': [],
            'step_actions': [],
            'step_gate_counts': [],
        }
        
        obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        """执行一步并记录数据"""
        obs, reward, done, info = self.env.step(action)
        
        # 记录step级别的数据
        self.current_episode['step_rewards'].append(float(reward))
        self.current_episode['step_fidelities'].append(float(info.get('fidelity', 0)))
        self.current_episode['step_actions'].append(int(action))
        self.current_episode['step_gate_counts'].append(int(info.get('gate_count', 0)))
        
        # 如果episode结束，保存数据
        if done:
            self._save_episode()
        
        return obs, reward, done, info
    
    def _save_episode(self):
        """保存一个完整的episode"""
        self.episode_count += 1
        self.total_episodes += 1
        
        # 计算episode统计信息
        total_reward = sum(self.current_episode['step_rewards'])
        length = len(self.current_episode['step_rewards'])
        final_fidelity = self.current_episode['step_fidelities'][-1] if self.current_episode['step_fidelities'] else 0
        success = final_fidelity >= 0.99
        
        # 构建episode记录
        episode_info = {
            'episode': self.total_episodes,
            'total_reward': total_reward,
            'length': length,
            'final_fidelity': final_fidelity,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'trajectory': dict(self.current_episode)
        }
        
        # 添加到批量缓存
        self.batch_cache.append(episode_info)
        
        # 如果达到batch_size，保存一次
        if len(self.batch_cache) >= self.batch_size:
            self._save_batch()
    
    def _save_batch(self):
        """批量保存数据"""
        if len(self.batch_cache) == 0:
            return
        
        # 保存CSV（追加模式）
        csv_records = []
        for ep in self.batch_cache:
            csv_records.append({
                'episode': ep['episode'],
                'total_reward': ep['total_reward'],
                'length': ep['length'],
                'final_fidelity': ep['final_fidelity'],
                'success': ep['success']
            })
        
        df = pd.DataFrame(csv_records)
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        
        # 保存JSONL（追加模式）
        with open(self.jsonl_path, 'a') as f:
            for ep in self.batch_cache:
                f.write(json.dumps(ep) + '\n')
        
        # 清空缓存
        print(f"💾 已保存{len(self.batch_cache)}个episodes (总计: {self.total_episodes})")
        self.batch_cache = []
    
    def save_final(self):
        """保存所有剩余数据和统计信息"""
        # 保存剩余的batch
        self._save_batch()
        
        # 计算并保存统计信息
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            if len(df) > 0:
                stats = {
                    'total_episodes': len(df),
                    'mean_reward': float(df['total_reward'].mean()),
                    'std_reward': float(df['total_reward'].std()),
                    'mean_fidelity': float(df['final_fidelity'].mean()),
                    'std_fidelity': float(df['final_fidelity'].std()),
                    'success_rate': float(df['success'].mean()),
                    'mean_length': float(df['length'].mean()),
                    'last_updated': datetime.now().isoformat(),
                }
                
                with open(self.stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"\n📊 最终统计:")
                print(f"  总episodes: {stats['total_episodes']}")
                print(f"  平均保真度: {stats['mean_fidelity']:.3f}")
                print(f"  成功率: {stats['success_rate']:.1%}")
                
                return stats
        
        return None
    
    def close(self):
        """关闭环境前保存所有数据"""
        self.save_final()
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __getattr__(self, name):
        """代理所有其他属性到内部环境"""
        return getattr(self.env, name)

