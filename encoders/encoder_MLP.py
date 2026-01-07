import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
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
    raise ValueError(f"Unknown activation: {name}")


class Encoder_MLP(nn.Module):
    """
    纯 MLP encoder（对标 CNN encoder）:
    gate_seq(token ids, padding=-1) -> one-hot(含 PAD 类) -> flatten -> MLP -> [B, out_dim]

    你的旧版是固定的 2 层 MLP + ReLU（hid=256 写死结构），这里只是把它参数化，便于 HPO 批量生成。
    """
    def __init__(
        self,
        actions,
        max_gates: int,
        hid: int = 256,
        *,
        out_dim: int | None = None,
        depth: int = 2,
        hidden_dims: list[int] | None = None,
        act: str = "relu",
        use_ln: bool = False,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.actions = actions
        self.max_gates = int(max_gates)

        # one-hot vocab size: A + 1(PAD)
        self.V = len(self.actions) + 1
        in_dim = self.max_gates * self.V

        out_dim = int(out_dim) if out_dim is not None else int(hid)
        self.out_dim = out_dim

        # ---- hidden dims 生成规则 ----
        # 1) 如果显式给 hidden_dims，就用它
        # 2) 否则按 depth 生成 [hid]*depth
        if hidden_dims is None:
            depth = int(depth)
            if depth < 1:
                raise ValueError("depth must be >= 1")
            hidden_dims = [int(hid)] * depth
        else:
            hidden_dims = [int(x) for x in hidden_dims]
            if len(hidden_dims) < 1:
                raise ValueError("hidden_dims must be non-empty")

        act_layer = _make_activation(act)
        do = float(dropout)
        in_do = float(input_dropout)

        layers: list[nn.Module] = []
        if in_do > 0:
            layers.append(nn.Dropout(p=in_do))

        # MLP: in_dim -> hidden_dims... -> out_dim
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            is_last = (i == len(dims) - 2)
            if not is_last:
                if use_ln:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_layer.__class__())  # 新实例，避免共享状态
                if do > 0:
                    layers.append(nn.Dropout(p=do))

        self.mlp = nn.Sequential(*layers)

    @property
    def num_params(self) -> int:
        """返回可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_cfg(cls, *, actions, max_gates: int, cfg: dict):
        """
        让 train.py 里直接传 net_cfg，就能构造 encoder。

        例子：
        net_cfg = {
          "hid": 256,
          "depth": 2,
          "act": "silu",
          "use_ln": True,
          "dropout": 0.1,
          "out_dim": 256,
          # 或者显式指定:
          # "hidden_dims": [256, 256, 256]
        }
        """
        cfg = dict(cfg or {})
        return cls(
            actions=actions,
            max_gates=max_gates,
            hid=int(cfg.get("hid", 256)),
            out_dim=cfg.get("out_dim", None),
            depth=int(cfg.get("depth", 2)),
            hidden_dims=cfg.get("hidden_dims", None),
            act=str(cfg.get("act", "relu")),
            use_ln=bool(cfg.get("use_ln", False)),
            dropout=float(cfg.get("dropout", 0.0)),
            input_dropout=float(cfg.get("input_dropout", 0.0)),
        )

    def forward(self, gate_seq) -> torch.Tensor:
        # 兼容 obs=dict 的情况：Encoder_MLP(obs) 也能工作
        if isinstance(gate_seq, dict) and "state" in gate_seq:
            gate_seq = gate_seq["state"]

        onehot = self.gate_seq_to_onehot(
            gate_seq,
            actions=self.actions,
            max_gates=self.max_gates,
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        feat = onehot.flatten(1)          # [B, Tmax*(Amax+1)]
        return self.mlp(feat)             # [B, out_dim]

    @staticmethod
    def gate_seq_to_onehot(
        gate_seq,
        actions,
        max_gates: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        根据 actions（动作表）生成 one-hot 编码，包含 padding 类。

        Args:
            gate_seq: [T] or [B,T] 的 action_id 序列，padding=-1
                    合法 action_id 必须在 [0, len(actions)-1]
            actions:   create_action_mapping(...) 生成的动作表(list)，len(actions)=A
            max_gates: 若提供则 pad/截断到该长度
            device:    输出设备
            dtype:     输出类型（默认 float32）

        Returns:
            onehot: [B, T, V]，其中 V = A + 1，最后一类是 PAD
        """
        A = len(actions)      # 动作数
        pad_id = A            # padding 的类别 id
        V = A + 1             # one-hot 维度（含 padding）

        # numpy -> torch
        if isinstance(gate_seq, np.ndarray):
            gate_seq = torch.from_numpy(gate_seq)

        # [T] -> [1,T]
        if gate_seq.dim() == 1:
            gate_seq = gate_seq.unsqueeze(0)

        # pad / truncate 到 max_gates
        if max_gates is not None:
            T = gate_seq.size(1)
            if T < max_gates:
                pad = gate_seq.new_full((gate_seq.size(0), max_gates - T), -1)
                gate_seq = torch.cat([gate_seq, pad], dim=1)
            elif T > max_gates:
                gate_seq = gate_seq[:, :max_gates]

        # device
        if device is None:
            device = gate_seq.device
        gate_seq = gate_seq.to(device)

        # -1 -> pad_id
        x = torch.where(gate_seq < 0, torch.full_like(gate_seq, pad_id), gate_seq).long()

        # 安全检查：避免出现越界 action_id
        if torch.any((x != pad_id) & ((x < 0) | (x >= A))):
            bad = x[(x != pad_id) & ((x < 0) | (x >= A))][:10].detach().cpu().tolist()
            raise ValueError(
                f"Found out-of-range action ids (expect 0..{A-1} or -1 padding). Examples: {bad}"
            )

        onehot = F.one_hot(x, num_classes=V).to(dtype=dtype)  # [B,T,V]
        return onehot


def build_encoder_mlp(*, actions, max_gates: int, net_cfg: dict | None = None) -> Encoder_MLP:
    """更偏工程化的工厂函数：train.py 里一行就能构造。"""
    return Encoder_MLP.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg or {})
