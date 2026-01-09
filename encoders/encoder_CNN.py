"""
CNN Encoder for quantum circuit state representations.

支持两种输入格式：
- 3d_tensor: [B, D=max_gates, N=n_qubits, C=n_channels]
- 2d_grid:   [B, 4, T=max_gates]

输出统一为 [B, out_dim]。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder_CNN(nn.Module):
    """
    CNN encoder 用于量子电路状态表示。
    
    支持两种输入：
    - 3d_tensor: [B, D=max_gates, N=n_qubits, C=n_channels]，空电路全 0
    - 2d_grid:   [B, 4, T=max_gates]，空电路前三行全 PAD_Q，第四行全 0
    
    输出：[B, out_dim]
    
    参数：
        out_dim: 输出维度（默认 256，仅在 use_proj=True 时生效）
        hid: 隐藏层通道数（默认 64）
        depth: CNN 层数（默认 3）
        kernel_size: 卷积核大小（默认 3）
        dropout: dropout 概率（默认 0.0）
        act: 激活函数（默认 "relu"）
        grid_encoding: grid 分支的编码方式，"scalar" 或 "onehot"（默认 "scalar"）
        mode: 输入模式，"auto" | "grid" | "tensor"（默认 "auto"）
        use_proj: 是否使用投影层（默认 True）
                  - True: 使用 Linear(hid -> out_dim) 投影，输出维度为 out_dim
                  - False: 不使用投影层，输出维度为 hid（由外部 sharedMLP 处理）
    """
    
    def __init__(
        self,
        *,
        out_dim: int = 256,
        hid: int = 64,
        depth: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
        act: str = "relu",
        grid_encoding: str = "scalar",
        mode: str = "auto",
        use_proj: bool = True,
    ):
        super().__init__()
        
        self.hid = int(hid)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.act_name = act
        self.grid_encoding = grid_encoding.lower()
        self.mode = mode.lower()
        self.use_proj = bool(use_proj)
        
        # 保存激活函数实例，用于 forward 中的 first conv 激活
        self.first_act = _make_activation(act)
        
        # out_dim 取决于是否使用投影层
        if self.use_proj:
            self.out_dim = int(out_dim)
        else:
            # 不使用投影层时，输出维度就是 hid
            self.out_dim = self.hid
        
        assert self.grid_encoding in ("scalar", "onehot"), \
            f"grid_encoding must be 'scalar' or 'onehot', got {grid_encoding}"
        assert self.mode in ("auto", "grid", "tensor"), \
            f"mode must be 'auto', 'grid', or 'tensor', got {mode}"
        
        # ========== Grid 分支 (1D Conv) ==========
        self._build_grid_branch()
        
        # ========== Tensor 分支 (2D Conv) ==========
        self._build_tensor_branch()
    
    def _build_grid_branch(self):
        """构建 grid 分支的 1D CNN"""
        act_layer = _make_activation(self.act_name)
        padding = self.kernel_size // 2  # 保持长度不变
        
        if self.grid_encoding == "scalar":
            # 直接用 4 个 channel（4 行作为 4 个 channel）
            self.grid_first = nn.Conv1d(
                in_channels=4,
                out_channels=self.hid,
                kernel_size=self.kernel_size,
                padding=padding,
            )
        else:
            # onehot 编码，用 LazyConv1d 让首层自动推断 in_channels
            self.grid_first = nn.LazyConv1d(
                out_channels=self.hid,
                kernel_size=self.kernel_size,
                padding=padding,
            )
        
        # 堆叠 depth 个 Conv1d blocks
        grid_layers = []
        for _ in range(self.depth):
            grid_layers.append(nn.Conv1d(
                in_channels=self.hid,
                out_channels=self.hid,
                kernel_size=self.kernel_size,
                padding=padding,
            ))
            grid_layers.append(act_layer.__class__())
            if self.dropout > 0:
                grid_layers.append(nn.Dropout(p=self.dropout))
        
        self.grid_backbone = nn.Sequential(*grid_layers)
        
        # 投影层（可选）
        if self.use_proj:
            self.grid_proj = nn.Linear(self.hid, self.out_dim)
        else:
            self.grid_proj = None
    
    def _build_tensor_branch(self):
        """构建 tensor 分支的 2D CNN"""
        act_layer = _make_activation(self.act_name)
        padding = self.kernel_size // 2  # 保持空间尺寸不变
        
        # 用 LazyConv2d 让首层自动推断 in_channels（因为 C 维度不固定）
        self.tensor_first = nn.LazyConv2d(
            out_channels=self.hid,
            kernel_size=self.kernel_size,
            padding=padding,
        )
        
        # 堆叠 depth 个 Conv2d blocks
        tensor_layers = []
        for _ in range(self.depth):
            tensor_layers.append(nn.Conv2d(
                in_channels=self.hid,
                out_channels=self.hid,
                kernel_size=self.kernel_size,
                padding=padding,
            ))
            tensor_layers.append(act_layer.__class__())
            if self.dropout > 0:
                tensor_layers.append(nn.Dropout(p=self.dropout))
        
        self.tensor_backbone = nn.Sequential(*tensor_layers)
        
        # 投影层（可选）
        if self.use_proj:
            self.tensor_proj = nn.Linear(self.hid, self.out_dim)
        else:
            self.tensor_proj = None

    @property
    def num_params(self) -> int:
        """
        返回可训练参数总数。
        
        注意：由于使用了 LazyConv，需要先调用一次 forward 初始化模型。
        此方法会跳过未初始化的 LazyModule 参数，只计算已初始化的部分。
        """
        total = 0
        has_uninitialized = False
        
        for p in self.parameters():
            if not p.requires_grad:
                continue
            try:
                total += p.numel()
            except ValueError:
                # UninitializedParameter from LazyModule
                has_uninitialized = True
        
        if has_uninitialized:
            import warnings
            warnings.warn(
                "CNN encoder has uninitialized LazyConv modules. "
                "The returned count only includes initialized parameters. "
                "Call forward() with appropriate input to initialize all branches."
            )
        
        return total

    @classmethod
    def from_cfg(cls, net_cfg: dict):
        """
        工厂方法，与其他 encoder 接口一致。
        
        例子：
        net_cfg = {
            "out_dim": 256,
            "hid": 64,
            "depth": 3,
            "kernel_size": 3,
            "dropout": 0.0,
            "act": "relu",
            "grid_encoding": "scalar",
            "mode": "auto",
            "use_proj": True,  # 是否使用投影层
        }
        """
        cfg = dict(net_cfg or {})
        return cls(
            out_dim=int(cfg.get("out_dim", 256)),
            hid=int(cfg.get("hid", 64)),
            depth=int(cfg.get("depth", 3)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.0)),
            act=str(cfg.get("act", "relu")),
            grid_encoding=str(cfg.get("grid_encoding", "scalar")),
            mode=str(cfg.get("mode", "auto")),
            use_proj=bool(cfg.get("use_proj", True)),
        )
    
    def _detect_input_type(self, x: torch.Tensor) -> str:
        """
        自动判断输入类型：grid 或 tensor。
        
        - grid: (4, T) 或 (B, 4, T)
        - tensor: (D, N, C) 或 (B, D, N, C)
        """
        shape = x.shape
        
        if len(shape) == 2:
            # (4, T) -> grid, (D, N) 不支持
            if shape[0] == 4:
                return "grid"
            else:
                raise ValueError(
                    f"2D input shape {shape} 不支持。"
                    f"Grid 输入应为 (4, T) 或 (B, 4, T)；"
                    f"Tensor 输入应为 (D, N, C) 或 (B, D, N, C)。"
                )
        
        elif len(shape) == 3:
            # (B, 4, T) -> grid
            # (D, N, C) -> tensor（无 batch，需要 unsqueeze）
            if shape[1] == 4:
                return "grid"
            else:
                return "tensor"
        
        elif len(shape) == 4:
            # (B, D, N, C) -> tensor
            return "tensor"
        
        else:
            raise ValueError(
                f"输入 shape {shape} 不支持。"
                f"Grid 输入应为 (4, T) 或 (B, 4, T)；"
                f"Tensor 输入应为 (D, N, C) 或 (B, D, N, C)。"
            )
    
    def forward(self, x) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入状态，可以是：
               - dict with "state" key
               - grid: (4, T) 或 (B, 4, T)
               - tensor: (D, N, C) 或 (B, D, N, C)
        
        Returns:
            [B, out_dim] 的特征向量
        """
        # 兼容 obs=dict 的情况
        if isinstance(x, dict) and "state" in x:
            x = x["state"]
        
        # numpy -> torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # 确保在正确设备上
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)
        
        # 确定输入类型
        if self.mode == "auto":
            input_type = self._detect_input_type(x)
        else:
            input_type = self.mode
        
        if input_type == "grid":
            return self._forward_grid(x)
        else:
            return self._forward_tensor(x)
    
    def _forward_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Grid 分支前向传播。
        
        输入: (4, T) 或 (B, 4, T)
        输出: [B, out_dim]
        """
        # 确保是 3D: [B, 4, T]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (4, T) -> (1, 4, T)
        
        B, _, T = x.shape
        
        # 推断 PAD_Q：前三行的最大值（因为 padding 就是 n_qubits）
        pad_q = int(x[:, 0:3, :].max().item())
        
        # 提取各行
        c0 = x[:, 0, :]  # [B, T] cnot_control
        c1 = x[:, 1, :]  # [B, T] cnot_target
        c2 = x[:, 2, :]  # [B, T] oneq_target
        gid = x[:, 3, :]  # [B, T] oneq_gate_id
        
        # 构造 valid_step mask（非 padding 的步骤）
        # padding 条件：c0==pad_q & c1==pad_q & c2==pad_q & gid==0
        pad_mask = (c0 == pad_q) & (c1 == pad_q) & (c2 == pad_q) & (gid == 0)
        valid = ~pad_mask  # [B, T]
        
        # 检查是否有有效步骤
        valid_count = valid.sum(dim=1)  # [B]
        all_empty = (valid_count == 0)
        
        # 如果全部是空电路，直接返回全 0
        if all_empty.all():
            return torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)
        
        # 编码
        if self.grid_encoding == "scalar":
            # 直接用 4 行作为 4 个 channel
            feat = x  # [B, 4, T]
        else:
            # onehot 编码
            feat = self._grid_onehot_encode(x, pad_q)  # [B, C_onehot, T]
        
        # Conv1d forward
        out = self.grid_first(feat)  # [B, hid, T]
        out = self.first_act(out)  # 使用配置的激活函数
        out = self.grid_backbone(out)  # [B, hid, T]
        
        # Masked mean pooling：只在 valid steps 上做平均
        # out: [B, hid, T], valid: [B, T]
        valid_mask = valid.unsqueeze(1).float()  # [B, 1, T]
        
        # 将无效位置的特征置 0
        out = out * valid_mask
        
        # 计算每个样本的平均
        valid_count_safe = valid_count.clamp(min=1).unsqueeze(1)  # [B, 1]
        pooled = out.sum(dim=2) / valid_count_safe  # [B, hid]
        
        # 对于完全空的序列，pooled 应该是 0
        if all_empty.any():
            pooled = torch.where(
                all_empty.unsqueeze(-1),
                torch.zeros_like(pooled),
                pooled
            )
        
        # 投影到 out_dim（如果启用）
        if self.use_proj:
            return self.grid_proj(pooled)  # [B, out_dim]
        else:
            return pooled  # [B, hid]
    
    def _grid_onehot_encode(self, x: torch.Tensor, pad_q: int) -> torch.Tensor:
        """
        将 grid 输入进行 onehot 编码。
        
        输入: [B, 4, T]
        输出: [B, C_onehot, T]
        
        编码：
        - 前三行（qubit indices）：onehot(size=pad_q+1)
        - 第四行（gate_id）：onehot(size=max_gate_id+1)
        """
        B, _, T = x.shape
        
        c0 = x[:, 0, :].long()  # [B, T] cnot_control
        c1 = x[:, 1, :].long()  # [B, T] cnot_target
        c2 = x[:, 2, :].long()  # [B, T] oneq_target
        gid = x[:, 3, :].long()  # [B, T] oneq_gate_id
        
        # qubit onehot size
        qubit_size = pad_q + 1
        
        # gate_id onehot size：找最大 gate_id
        max_gid = max(int(gid.max().item()), 1)
        gid_size = max_gid + 1
        
        # 生成 onehot
        # F.one_hot 输出 [B, T, C]，需要 permute 成 [B, C, T]
        c0_oh = F.one_hot(c0.clamp(0, qubit_size - 1), num_classes=qubit_size)  # [B, T, qubit_size]
        c1_oh = F.one_hot(c1.clamp(0, qubit_size - 1), num_classes=qubit_size)  # [B, T, qubit_size]
        c2_oh = F.one_hot(c2.clamp(0, qubit_size - 1), num_classes=qubit_size)  # [B, T, qubit_size]
        gid_oh = F.one_hot(gid.clamp(0, gid_size - 1), num_classes=gid_size)     # [B, T, gid_size]
        
        # 拼接并 permute
        concat = torch.cat([c0_oh, c1_oh, c2_oh, gid_oh], dim=-1)  # [B, T, total_C]
        return concat.permute(0, 2, 1).float()  # [B, total_C, T]
    
    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tensor 分支前向传播。
        
        输入: (D, N, C) 或 (B, D, N, C)
        输出: [B, out_dim]
        """
        # 确保是 4D: [B, D, N, C]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (D, N, C) -> (1, D, N, C)
        
        B, D, N, C = x.shape
        
        # 构造 valid_depth mask：每个 depth 是否有非零值
        # valid_d[b, d] = (x[b, d, :, :].abs().sum() > 0)
        valid_d = (x.abs().sum(dim=(2, 3)) > 0)  # [B, D]
        
        # 检查是否有有效 depth
        valid_count = valid_d.sum(dim=1)  # [B]
        all_empty = (valid_count == 0)
        
        # 如果全部是空电路，直接返回全 0
        if all_empty.all():
            return torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)
        
        # permute 成 [B, C, D, N] 用于 Conv2d
        x_perm = x.permute(0, 3, 1, 2)  # [B, C, D, N]
        
        # Conv2d forward
        out = self.tensor_first(x_perm)  # [B, hid, D, N]
        out = self.first_act(out)  # 使用配置的激活函数
        out = self.tensor_backbone(out)  # [B, hid, D, N]
        
        # Masked pooling：
        # 1. 根据 valid_d mask 将无效 depth 置 0
        # 2. 对 N 维度做平均
        # 3. 对 D 维度做 masked mean
        
        # valid_d: [B, D] -> [B, 1, D, 1] 用于 broadcast
        valid_mask = valid_d.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, D, 1]
        
        # 将无效 depth 的特征置 0
        out = out * valid_mask  # [B, hid, D, N]
        
        # 对 N 维度做平均
        out = out.mean(dim=3)  # [B, hid, D]
        
        # 对 D 维度做 masked mean
        valid_count_safe = valid_count.clamp(min=1).unsqueeze(1)  # [B, 1]
        pooled = out.sum(dim=2) / valid_count_safe  # [B, hid]
        
        # 对于完全空的序列，pooled 应该是 0
        if all_empty.any():
            pooled = torch.where(
                all_empty.unsqueeze(-1),
                torch.zeros_like(pooled),
                pooled
            )
        
        # 投影到 out_dim（如果启用）
        if self.use_proj:
            return self.tensor_proj(pooled)  # [B, out_dim]
        else:
            return pooled  # [B, hid]


def build_encoder_cnn(actions, max_gates: int, net_cfg: dict | None = None) -> Encoder_CNN:
    """
    工厂函数：与其他 encoder 工厂函数接口一致。
    
    注意：actions 和 max_gates 参数保留以保持接口一致性，
    但 CNN encoder 不依赖这些参数（它从输入动态推断）。
    """
    return Encoder_CNN.from_cfg(net_cfg or {})


if __name__ == "__main__":
    """
    Sanity check:
    - Grid input [2, 4, 10]
    - Tensor input [2, 10, 3, 7]
    - Empty circuit cases
    - Check output shape and finite values (no NaN/Inf)
    """
    print("=" * 60)
    print("Encoder_CNN Sanity Check")
    print("=" * 60)
    
    # Create encoder with default config
    cfg = {
        "out_dim": 256,
        "hid": 64,
        "depth": 3,
        "kernel_size": 3,
        "grid_encoding": "scalar",
        "mode": "auto",
    }
    encoder = Encoder_CNN.from_cfg(cfg)
    print(f"\n[Config] out_dim={encoder.out_dim}, hid={encoder.hid}, depth={encoder.depth}")
    print(f"[Config] kernel_size={encoder.kernel_size}, grid_encoding={encoder.grid_encoding}, mode={encoder.mode}")
    
    # ========== Grid Branch Test ==========
    print("\n" + "=" * 40)
    print("Grid Branch Test")
    print("=" * 40)
    
    # Simulate n_qubits=3, max_gates=10
    n_qubits = 3
    PAD_Q = n_qubits
    
    # Test 1: Normal grid input [2, 4, 10]
    print("\n--- Test 1: Grid input shape [2, 4, 10] ---")
    grid_input = np.full((2, 4, 10), PAD_Q, dtype=np.float32)
    grid_input[:, 3, :] = 0  # gate_id row = 0
    
    # Sample 0: 3 valid steps
    grid_input[0, 0, 0] = 0; grid_input[0, 1, 0] = 1  # CNOT(0, 1)
    grid_input[0, 2, 1] = 2; grid_input[0, 3, 1] = 1  # H on qubit 2
    grid_input[0, 0, 2] = 1; grid_input[0, 1, 2] = 2  # CNOT(1, 2)
    
    # Sample 1: 5 valid steps
    for t in range(5):
        grid_input[1, 2, t] = t % n_qubits
        grid_input[1, 3, t] = (t % 5) + 1  # gate_id 1-5
    
    print(f"Input shape: {grid_input.shape}")
    out_grid = encoder(grid_input)
    print(f"Output shape: {out_grid.shape}")
    assert out_grid.shape == (2, 256), f"Expected (2, 256), got {out_grid.shape}"
    assert torch.isfinite(out_grid).all(), "Output contains NaN/Inf!"
    print("[PASS] Test 1")
    
    # Test 2: Empty circuit (all padding)
    print("\n--- Test 2: Grid empty circuit ---")
    empty_grid = np.full((2, 4, 10), PAD_Q, dtype=np.float32)
    empty_grid[:, 3, :] = 0  # gate_id row = 0
    
    print(f"Input shape: {empty_grid.shape}")
    out_empty_grid = encoder(empty_grid)
    print(f"Output shape: {out_empty_grid.shape}")
    assert out_empty_grid.shape == (2, 256), f"Expected (2, 256), got {out_empty_grid.shape}"
    assert torch.isfinite(out_empty_grid).all(), "Output contains NaN/Inf!"
    assert (out_empty_grid == 0).all(), "Empty circuit should output all zeros!"
    print("[PASS] Test 2 (empty circuit outputs zeros)")
    
    # Test 3: Single grid input [4, 10]
    print("\n--- Test 3: Grid single input shape [4, 10] ---")
    single_grid = grid_input[0]  # [4, 10]
    print(f"Input shape: {single_grid.shape}")
    out_single = encoder(single_grid)
    print(f"Output shape: {out_single.shape}")
    assert out_single.shape == (1, 256), f"Expected (1, 256), got {out_single.shape}"
    print("[PASS] Test 3")
    
    # ========== Tensor Branch Test ==========
    print("\n" + "=" * 40)
    print("Tensor Branch Test")
    print("=" * 40)
    
    # Test 4: Normal tensor input [2, 10, 3, 7]
    print("\n--- Test 4: Tensor input shape [2, 10, 3, 7] ---")
    tensor_input = np.zeros((2, 10, 3, 7), dtype=np.float32)
    
    # Sample 0: 3 valid depths
    tensor_input[0, 0, 0, 0] = 1.0  # H on qubit 0
    tensor_input[0, 1, 1, 5] = 1.0  # CNOT_control on qubit 1
    tensor_input[0, 1, 2, 6] = 1.0  # CNOT_target on qubit 2
    tensor_input[0, 2, 2, 1] = 1.0  # X on qubit 2
    
    # Sample 1: 5 valid depths
    for d in range(5):
        tensor_input[1, d, d % 3, d % 7] = 1.0
    
    print(f"Input shape: {tensor_input.shape}")
    out_tensor = encoder(tensor_input)
    print(f"Output shape: {out_tensor.shape}")
    assert out_tensor.shape == (2, 256), f"Expected (2, 256), got {out_tensor.shape}"
    assert torch.isfinite(out_tensor).all(), "Output contains NaN/Inf!"
    print("[PASS] Test 4")
    
    # Test 5: Empty circuit (all zeros)
    print("\n--- Test 5: Tensor empty circuit ---")
    empty_tensor = np.zeros((2, 10, 3, 7), dtype=np.float32)
    
    print(f"Input shape: {empty_tensor.shape}")
    out_empty_tensor = encoder(empty_tensor)
    print(f"Output shape: {out_empty_tensor.shape}")
    assert out_empty_tensor.shape == (2, 256), f"Expected (2, 256), got {out_empty_tensor.shape}"
    assert torch.isfinite(out_empty_tensor).all(), "Output contains NaN/Inf!"
    assert (out_empty_tensor == 0).all(), "Empty circuit should output all zeros!"
    print("[PASS] Test 5 (empty circuit outputs zeros)")
    
    # Test 6: Single tensor input [10, 3, 7]
    print("\n--- Test 6: Tensor single input shape [10, 3, 7] ---")
    single_tensor = tensor_input[0]  # [10, 3, 7]
    print(f"Input shape: {single_tensor.shape}")
    out_single_tensor = encoder(single_tensor)
    print(f"Output shape: {out_single_tensor.shape}")
    assert out_single_tensor.shape == (1, 256), f"Expected (1, 256), got {out_single_tensor.shape}"
    print("[PASS] Test 6")
    
    # ========== Dict Input Compatibility Test ==========
    print("\n" + "=" * 40)
    print("Dict Input Compatibility Test")
    print("=" * 40)
    
    print("\n--- Test 7: Dict input ---")
    obs_dict = {"state": grid_input, "action_mask": None}
    out_dict = encoder(obs_dict)
    print(f"Input: dict with 'state' key")
    print(f"Output shape: {out_dict.shape}")
    assert out_dict.shape == (2, 256), f"Expected (2, 256), got {out_dict.shape}"
    print("[PASS] Test 7")
    
    # ========== Onehot Encoding Test ==========
    print("\n" + "=" * 40)
    print("Onehot Encoding Test")
    print("=" * 40)
    
    print("\n--- Test 8: grid_encoding='onehot' ---")
    cfg_onehot = {
        "out_dim": 128,
        "hid": 32,
        "depth": 2,
        "kernel_size": 3,
        "grid_encoding": "onehot",
        "mode": "grid",
    }
    encoder_onehot = Encoder_CNN.from_cfg(cfg_onehot)
    out_onehot = encoder_onehot(grid_input)
    print(f"Output shape: {out_onehot.shape}")
    assert out_onehot.shape == (2, 128), f"Expected (2, 128), got {out_onehot.shape}"
    assert torch.isfinite(out_onehot).all(), "Output contains NaN/Inf!"
    print("[PASS] Test 8")
    
    # ========== Factory Function Test ==========
    print("\n" + "=" * 40)
    print("Factory Function Test")
    print("=" * 40)
    
    print("\n--- Test 9: build_encoder_cnn ---")
    actions = [f"gate_{i}" for i in range(32)]  # mock actions
    encoder_factory = build_encoder_cnn(actions, max_gates=20, net_cfg={"out_dim": 64})
    print(f"out_dim: {encoder_factory.out_dim}")
    assert encoder_factory.out_dim == 64
    out_factory = encoder_factory(grid_input)
    print(f"Output shape: {out_factory.shape}")
    assert out_factory.shape == (2, 64), f"Expected (2, 64), got {out_factory.shape}"
    print("[PASS] Test 9")
    
    # ========== use_proj=False Test ==========
    print("\n" + "=" * 40)
    print("use_proj=False Test")
    print("=" * 40)
    
    print("\n--- Test 10: use_proj=False (Grid) ---")
    cfg_no_proj = {
        "hid": 128,
        "depth": 2,
        "use_proj": False,  # no projection layer
    }
    encoder_no_proj = Encoder_CNN.from_cfg(cfg_no_proj)
    print(f"use_proj: {encoder_no_proj.use_proj}")
    print(f"hid: {encoder_no_proj.hid}")
    print(f"out_dim: {encoder_no_proj.out_dim}")
    assert encoder_no_proj.out_dim == 128, f"Expected out_dim=hid=128, got {encoder_no_proj.out_dim}"
    assert encoder_no_proj.grid_proj is None, "grid_proj should be None when use_proj=False"
    assert encoder_no_proj.tensor_proj is None, "tensor_proj should be None when use_proj=False"
    
    out_no_proj_grid = encoder_no_proj(grid_input)
    print(f"Grid output shape: {out_no_proj_grid.shape}")
    assert out_no_proj_grid.shape == (2, 128), f"Expected (2, 128), got {out_no_proj_grid.shape}"
    print("[PASS] Test 10")
    
    print("\n--- Test 11: use_proj=False (Tensor) ---")
    out_no_proj_tensor = encoder_no_proj(tensor_input)
    print(f"Tensor output shape: {out_no_proj_tensor.shape}")
    assert out_no_proj_tensor.shape == (2, 128), f"Expected (2, 128), got {out_no_proj_tensor.shape}"
    print("[PASS] Test 11")
    
    print("\n--- Test 12: use_proj=False (Empty circuit) ---")
    out_no_proj_empty = encoder_no_proj(empty_grid)
    print(f"Empty circuit output shape: {out_no_proj_empty.shape}")
    assert out_no_proj_empty.shape == (2, 128), f"Expected (2, 128), got {out_no_proj_empty.shape}"
    assert (out_no_proj_empty == 0).all(), "Empty circuit should output all zeros!"
    print("[PASS] Test 12")
    
    print("\n" + "=" * 60)
    print("All sanity checks passed!")
    print("=" * 60)
