import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _make_activation(name: str) -> nn.Module:
    """复制自 encoder_MLP.py，保持风格一致"""
    name = (name or "relu").lower()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class Encoder_RNN(nn.Module):
    """
    RNN encoder for gate_seq state scheme:
    gate_seq(token ids, padding=-1) -> embedding -> RNN -> pool -> proj -> [B, out_dim]
    
    接口与 Encoder_MLP 完全兼容，支持:
    - from_cfg(actions, max_gates, cfg) 工厂方法
    - forward 接受 [T]/[B,T] 的 gate token ids (padding=-1)
    - 空电路（全 padding）不报错
    """
    
    def __init__(
        self,
        actions,
        max_gates: int,
        *,
        embed_dim: int = 64,
        pre_dim: int | None = None,
        pre_norm: str = "ln",
        pre_act: str = "relu",
        rnn_type: str = "lstm",
        hidden_size: int = 100,
        num_layers: int = 1,
        bidirectional: bool = False,
        rnn_dropout: float = 0.0,
        pool: str = "last",
        out_dim: int | None = None,
        post_ln: bool = False,
        post_act: str = "none",
        post_dropout: float = 0.0,
    ):
        super().__init__()
        self.actions = actions
        self.max_gates = int(max_gates)
        
        # vocab size: A + 1 (PAD)
        A = len(actions)
        self.pad_id = A
        self.vocab_size = A + 1
        
        # pooling 方式
        assert pool in ("last", "mean"), f"pool must be 'last' or 'mean', got {pool}"
        self.pool = pool
        
        # RNN 类型
        rnn_type = rnn_type.lower()
        assert rnn_type in ("lstm", "gru"), f"rnn_type must be 'lstm' or 'gru', got {rnn_type}"
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=self.pad_id,
        )
        
        # 可选的 pre-coding layer (embed -> pre_dim with norm + act)
        rnn_input_dim = embed_dim
        if pre_dim is not None:
            pre_layers = [nn.Linear(embed_dim, pre_dim)]
            if pre_norm == "ln":
                pre_layers.append(nn.LayerNorm(pre_dim))
            elif pre_norm == "bn":
                # BatchNorm1d 需要 (B, C, T) 或 (B*T, C)，这里用 LayerNorm 替代更方便
                pre_layers.append(nn.LayerNorm(pre_dim))  # 实际上对序列数据 LN 更常用
            elif pre_norm != "none":
                raise ValueError(f"pre_norm must be 'ln', 'bn', or 'none', got {pre_norm}")
            pre_layers.append(_make_activation(pre_act))
            self.pre_coding = nn.Sequential(*pre_layers)
            rnn_input_dim = pre_dim
        else:
            self.pre_coding = None
        
        # RNN layer
        RNN_CLS = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = RNN_CLS(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN 输出维度
        rnn_out_dim = hidden_size * self.num_directions
        
        # 输出维度
        if out_dim is None:
            out_dim = rnn_out_dim
        self.out_dim = out_dim
        
        # Projection layer (rnn_out_dim -> out_dim)
        if out_dim != rnn_out_dim:
            self.proj = nn.Linear(rnn_out_dim, out_dim)
        else:
            self.proj = None
        
        # 可选的 post processing
        post_layers = []
        if post_ln:
            post_layers.append(nn.LayerNorm(out_dim))
        if post_act and post_act.lower() != "none":
            post_layers.append(_make_activation(post_act))
        if post_dropout > 0:
            post_layers.append(nn.Dropout(p=post_dropout))
        
        self.post = nn.Sequential(*post_layers) if post_layers else None

    @property
    def num_params(self) -> int:
        """返回可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_cfg(cls, *, actions, max_gates: int, cfg: dict):
        """
        工厂方法，与 Encoder_MLP.from_cfg 接口一致。
        
        例子：
        cfg = {
            "embed_dim": 64,
            "pre_dim": 128,
            "pre_norm": "ln",
            "pre_act": "relu",
            "rnn_type": "lstm",
            "hidden_size": 100,
            "num_layers": 1,
            "bidirectional": False,
            "rnn_dropout": 0.0,
            "pool": "last",
            "out_dim": 256,
            "post_ln": False,
            "post_act": "relu",
            "post_dropout": 0.0,
        }
        """
        cfg = dict(cfg or {})
        return cls(
            actions=actions,
            max_gates=max_gates,
            embed_dim=int(cfg.get("embed_dim", 64)),
            pre_dim=cfg.get("pre_dim", None),
            pre_norm=str(cfg.get("pre_norm", "ln")),
            pre_act=str(cfg.get("pre_act", "relu")),
            rnn_type=str(cfg.get("rnn_type", "lstm")),
            hidden_size=int(cfg.get("hidden_size", 100)),
            num_layers=int(cfg.get("num_layers", 1)),
            bidirectional=bool(cfg.get("bidirectional", False)),
            rnn_dropout=float(cfg.get("rnn_dropout", 0.0)),
            pool=str(cfg.get("pool", "last")),
            out_dim=cfg.get("out_dim", None),
            post_ln=bool(cfg.get("post_ln", False)),
            post_act=str(cfg.get("post_act", "relu")),
            post_dropout=float(cfg.get("post_dropout", 0.0)),
        )
    
    def forward(self, gate_seq) -> torch.Tensor:
        """
        Args:
            gate_seq: [T] or [B, T] 的 action_id 序列，padding=-1
                     也可以是 dict with "state" key（兼容 Encoder_MLP）
        
        Returns:
            [B, out_dim] 的特征向量
        """
        # 兼容 obs=dict 的情况
        if isinstance(gate_seq, dict) and "state" in gate_seq:
            gate_seq = gate_seq["state"]
        
        device = next(self.parameters()).device
        
        # 转换为 tensor
        x = self._prepare_input(gate_seq, device)  # [B, T] long
        
        B, T = x.shape
        
        # 计算每条序列的有效长度（非 pad 的 token 数）
        lengths = (x != self.pad_id).sum(dim=1)  # [B]
        
        # 空序列处理：clamp 到最小 1，避免 pack 崩溃
        lengths_clamped = lengths.clamp(min=1)
        
        # Embedding
        emb = self.embed(x)  # [B, T, embed_dim]
        
        # 可选的 pre-coding layer
        if self.pre_coding is not None:
            emb = self.pre_coding(emb)  # [B, T, pre_dim]
        
        # Pack -> RNN -> Unpack
        packed = pack_padded_sequence(
            emb, 
            lengths_clamped.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        if self.rnn_type == "lstm":
            packed_out, (h_n, c_n) = self.rnn(packed)
        else:  # gru
            packed_out, h_n = self.rnn(packed)
        
        # Pooling
        if self.pool == "last":
            # 使用 h_n：[num_layers * num_directions, B, hidden_size]
            # 取最后一层的 hidden state
            if self.bidirectional:
                # 拼接最后一层的前向和后向 hidden
                h_fwd = h_n[-2]  # [B, hidden_size]
                h_bwd = h_n[-1]  # [B, hidden_size]
                pooled = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, hidden_size * 2]
            else:
                pooled = h_n[-1]  # [B, hidden_size]
        else:  # mean pooling
            # Unpack 输出
            output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            # output: [B, T, hidden_size * num_directions]
            
            # 创建 mask
            mask = (x != self.pad_id).unsqueeze(-1).float()  # [B, T, 1]
            
            # 对有效位置求和，除以有效长度
            summed = (output * mask).sum(dim=1)  # [B, hidden_size * num_directions]
            pooled = summed / lengths_clamped.unsqueeze(-1).float()  # [B, hidden_size * num_directions]
        
        # 对于完全空的序列（原始 lengths=0），pooled 可能包含 NaN
        # 用 0 替换（因为这些序列没有任何有效 token）
        if (lengths == 0).any():
            pooled = torch.where(
                lengths.unsqueeze(-1) == 0,
                torch.zeros_like(pooled),
                pooled
            )
        
        # Projection
        if self.proj is not None:
            out = self.proj(pooled)
        else:
            out = pooled
        
        # Post processing
        if self.post is not None:
            out = self.post(out)
        
        return out  # [B, out_dim]
    
    def _prepare_input(self, gate_seq, device: torch.device) -> torch.Tensor:
        """
        将输入转换为 [B, T] 的 long tensor，处理 padding 和越界检查。
        """
        # numpy -> torch
        if isinstance(gate_seq, np.ndarray):
            gate_seq = torch.from_numpy(gate_seq)
        
        # [T] -> [1, T]
        if gate_seq.dim() == 1:
            gate_seq = gate_seq.unsqueeze(0)
        
        # pad / truncate 到 max_gates
        T = gate_seq.size(1)
        if T < self.max_gates:
            pad = gate_seq.new_full((gate_seq.size(0), self.max_gates - T), -1)
            gate_seq = torch.cat([gate_seq, pad], dim=1)
        elif T > self.max_gates:
            gate_seq = gate_seq[:, :self.max_gates]
        
        # 移动到目标设备
        gate_seq = gate_seq.to(device)
        
        # -1 -> pad_id
        A = self.vocab_size - 1  # 实际动作数
        x = torch.where(gate_seq < 0, torch.full_like(gate_seq, self.pad_id), gate_seq).long()
        
        # 安全检查：非 pad 的 token 必须在 [0, A-1]
        invalid_mask = (x != self.pad_id) & ((x < 0) | (x >= A))
        if invalid_mask.any():
            bad = x[invalid_mask][:10].detach().cpu().tolist()
            raise ValueError(
                f"Found out-of-range action ids (expect 0..{A-1} or -1 padding). Examples: {bad}"
            )
        
        return x  # [B, max_gates] long


def build_encoder_rnn(actions, max_gates: int, net_cfg: dict | None = None) -> Encoder_RNN:
    """工厂函数：train.py 里一行就能构造。"""
    return Encoder_RNN.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg or {})


if __name__ == "__main__":
    """
    最小自检：
    - A=32, max_gates=60
    - 测试 [60] 输入 => [1, out_dim]
    - 测试 [4, 60] 输入（含全 padding 行） => [4, out_dim]
    """
    print("=" * 60)
    print("Encoder_RNN 自检测试")
    print("=" * 60)
    
    # 模拟 actions（32 个动作）
    A = 32
    actions = [f"gate_{i}" for i in range(A)]
    max_gates = 60
    
    # 创建 encoder（使用默认配置）
    cfg = {
        "embed_dim": 64,
        "rnn_type": "lstm",
        "hidden_size": 100,
        "num_layers": 1,
        "bidirectional": False,
        "pool": "last",
        "out_dim": 256,
    }
    encoder = Encoder_RNN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg)
    print(f"\n[Config] out_dim={encoder.out_dim}, vocab_size={encoder.vocab_size}, pad_id={encoder.pad_id}")
    print(f"[Config] rnn_type={encoder.rnn_type}, hidden_size={encoder.hidden_size}, pool={encoder.pool}")
    
    # 测试 1: 单条序列 [60]（含 -1 padding）
    print("\n--- Test 1: 输入 shape [60] ---")
    gate_seq_1d = np.concatenate([
        np.random.randint(0, A, size=30),  # 30 个有效 token
        np.full(30, -1)                     # 30 个 padding
    ])
    print(f"输入 shape: {gate_seq_1d.shape}")
    out_1d = encoder(gate_seq_1d)
    print(f"输出 shape: {out_1d.shape}")
    assert out_1d.shape == (1, 256), f"Expected (1, 256), got {out_1d.shape}"
    print("✓ Test 1 通过")
    
    # 测试 2: batch 输入 [4, 60]（含一个全 padding 行）
    print("\n--- Test 2: 输入 shape [4, 60]（含全 padding 行）---")
    gate_seq_2d = np.full((4, 60), -1, dtype=np.int64)
    gate_seq_2d[0, :20] = np.random.randint(0, A, size=20)  # 第 0 行有 20 个有效
    gate_seq_2d[1, :40] = np.random.randint(0, A, size=40)  # 第 1 行有 40 个有效
    gate_seq_2d[2, :5] = np.random.randint(0, A, size=5)    # 第 2 行有 5 个有效
    # 第 3 行全是 -1（全 padding，空电路）
    print(f"输入 shape: {gate_seq_2d.shape}")
    print(f"各行有效长度: {[(row != -1).sum() for row in gate_seq_2d]}")
    out_2d = encoder(gate_seq_2d)
    print(f"输出 shape: {out_2d.shape}")
    assert out_2d.shape == (4, 256), f"Expected (4, 256), got {out_2d.shape}"
    assert not torch.isnan(out_2d).any(), "Output contains NaN!"
    print("✓ Test 2 通过（全 padding 行不报错）")
    
    # 测试 3: dict 输入兼容性
    print("\n--- Test 3: dict 输入兼容性 ---")
    obs_dict = {"state": gate_seq_2d, "other_key": None}
    out_dict = encoder(obs_dict)
    print(f"输入: dict with 'state' key, shape {gate_seq_2d.shape}")
    print(f"输出 shape: {out_dict.shape}")
    assert out_dict.shape == (4, 256), f"Expected (4, 256), got {out_dict.shape}"
    print("✓ Test 3 通过")
    
    # 测试 4: bidirectional + mean pooling
    print("\n--- Test 4: bidirectional + mean pooling ---")
    cfg_bi = {
        "embed_dim": 32,
        "rnn_type": "gru",
        "hidden_size": 50,
        "num_layers": 2,
        "bidirectional": True,
        "pool": "mean",
        "out_dim": 128,
    }
    encoder_bi = Encoder_RNN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg_bi)
    out_bi = encoder_bi(gate_seq_2d)
    print(f"配置: bidirectional=True, pool='mean', out_dim=128")
    print(f"输出 shape: {out_bi.shape}")
    assert out_bi.shape == (4, 128), f"Expected (4, 128), got {out_bi.shape}"
    assert not torch.isnan(out_bi).any(), "Output contains NaN!"
    print("✓ Test 4 通过")
    
    # 测试 5: 通过 build_encoder_rnn 工厂函数
    print("\n--- Test 5: build_encoder_rnn 工厂函数 ---")
    encoder_factory = build_encoder_rnn(actions, max_gates, net_cfg={"out_dim": 64})
    out_factory = encoder_factory(gate_seq_1d)
    print(f"输出 shape: {out_factory.shape}")
    assert out_factory.shape == (1, 64), f"Expected (1, 64), got {out_factory.shape}"
    print("✓ Test 5 通过")
    
    print("\n" + "=" * 60)
    print("所有自检测试通过！")
    print("=" * 60)
