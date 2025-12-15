import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_MLP(nn.Module):
    """
    纯 MLP encoder（对标 CNN encoder）:
    token -> one-hot(固定) -> flatten -> MLP -> [B,hid]
    """
    def __init__(self, actions,max_gates: int, hid: int = 256):
        super().__init__()
        self.actions = actions
        self.max_gates = max_gates
        self.V = len(self.actions) + 1
        in_dim = max_gates * self.V

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )

    def forward(self, gate_seq) -> torch.Tensor:
        onehot = self.gate_seq_to_onehot(
            gate_seq,
            actions=self.actions,
            max_gates=self.max_gates,
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        feat = onehot.flatten(1)          # [B, Tmax*(Amax+1)]
        return self.mlp(feat)             # [B, hid]
        
    def gate_seq_to_onehot(self, 
                           gate_seq,   
                           actions,
                           max_gates: int | None = None, 
                           device: torch.device | None = None, 
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
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

        # （可选）安全检查：避免出现越界 action_id
        if torch.any((x != pad_id) & ((x < 0) | (x >= A))):
            bad = x[(x != pad_id) & ((x < 0) | (x >= A))][:10].detach().cpu().tolist()
            raise ValueError(f"Found out-of-range action ids (expect 0..{A-1} or -1 padding). Examples: {bad}")

        onehot = F.one_hot(x, num_classes=V).to(dtype=dtype)  # [B,T,V]
        return onehot