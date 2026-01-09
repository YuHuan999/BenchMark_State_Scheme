"""
Encoder_GIN: Graph Isomorphism Network encoder for graph-based state representations.

输入：图结构 (X: 节点特征, A: 邻接矩阵, mask: 节点有效掩码)
输出：[B, out_dim] 的图级特征向量

接口与 Encoder_MLP / Encoder_RNN 完全对齐：
  - self.out_dim 必须存在
  - @classmethod from_cfg(cls, *, actions, max_gates: int, cfg: dict)
  - forward(self, x) 返回 [B, out_dim]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
    """与 encoder_MLP.py 保持一致的激活函数构造器"""
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


def _make_norm(name: str, dim: int) -> nn.Module:
    """构造归一化层"""
    name = (name or "none").lower()
    if name in ("ln", "layernorm"):
        return nn.LayerNorm(dim)
    if name in ("bn", "batchnorm"):
        # 对于图节点特征，使用 BatchNorm1d 需要 reshape
        # 这里用 LayerNorm 替代，更适合变长序列
        return nn.LayerNorm(dim)
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")


class GINLayer(nn.Module):
    """
    单层 GIN (Graph Isomorphism Network) 层：
    
    H' = MLP((1 + eps) * H + A @ H)
    
    其中：
    - H: 节点特征 [B, N, hid]
    - A: 邻接矩阵 [B, N, N]
    - eps: 可学习或固定的标量
    - MLP: 多层感知机
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_depth: int = 2,
        mlp_hid: int | None = None,
        eps_learnable: bool = True,
        norm: str = "ln",
        dropout: float = 0.0,
        act: str = "relu",  # 激活函数，支持 relu/silu/gelu 等
    ):
        super().__init__()
        
        mlp_hid = mlp_hid if mlp_hid is not None else out_dim
        
        # epsilon 参数
        if eps_learnable:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))
        
        # 构建 MLP
        # 结构：Linear -> Norm -> Act (重复 mlp_depth-1 次) -> Linear
        layers = []
        dims = [in_dim] + [mlp_hid] * (mlp_depth - 1) + [out_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers.append(_make_norm(norm, dims[i + 1]))
                layers.append(_make_activation(act))  # 使用配置的激活函数
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: 节点特征 [B, N, in_dim]
            A: 邻接矩阵 [B, N, N]
        
        Returns:
            H': 更新后的节点特征 [B, N, out_dim]
        """
        # 消息传递：A @ H
        Ah = torch.bmm(A, H)  # [B, N, in_dim]
        
        # 聚合：(1 + eps) * H + Ah
        S = (1 + self.eps) * H + Ah  # [B, N, in_dim]
        
        # MLP 变换
        H_out = self.mlp(S)  # [B, N, out_dim]
        
        return H_out


class Encoder_GIN(nn.Module):
    """
    GIN (Graph Isomorphism Network) Encoder:
    
    图结构输入 -> GIN layers -> graph pooling -> 输出向量 [B, out_dim]
    
    支持两种输入格式：
    1. dict 格式: {"x": X, "a": A, "mask": M (optional)}
    2. tuple/list 格式: (X, A) 或 (X, A, M)
    
    其中：
    - X: [B, N, F] 或 [N, F] 节点特征
    - A: [B, N, N] 或 [N, N] 邻接矩阵
    - M: [B, N] 或 [N] 节点有效掩码（可选，默认全 1）
    """
    
    def __init__(
        self,
        actions,
        max_gates: int,
        *,
        out_dim: int = 256,
        hid: int = 128,
        depth: int = 3,
        mlp_depth: int = 2,
        mlp_hid: int | None = None,
        eps_learnable: bool = True,
        readout: str = "mean",
        use_undirected: bool = True,
        add_self_loops: bool = True,
        norm: str = "ln",
        act: str = "relu",  # GIN 内部 MLP 的激活函数
        dropout: float = 0.0,
        use_post_block: bool = True,
        post_ln: bool = False,
        post_act: str = "relu",
        post_dropout: float = 0.0,
        n_qubits: int | None = None,
    ):
        super().__init__()
        
        self.actions = actions
        self.max_gates = int(max_gates)
        self.hid = int(hid)
        self.depth = int(depth)
        self.readout = readout.lower()
        self.use_undirected = bool(use_undirected)
        self.add_self_loops = bool(add_self_loops)
        self.use_post_block = bool(use_post_block)
        
        # ===== Flat vector unpacking 参数（与 RepresentationWrapper.graph 对齐）=====
        # 如果 n_qubits 未指定，尝试从 actions 推断
        if n_qubits is None:
            # 从 actions 中推断 n_qubits（找最大的 target/control 值）
            max_qubit = 0
            for a in (actions or []):
                if isinstance(a, dict):
                    max_qubit = max(max_qubit, a.get("target", 0) or 0)
                    max_qubit = max(max_qubit, a.get("control", 0) or 0)
            n_qubits = max_qubit + 1 if max_qubit > 0 else 4  # 默认 4 qubits
        
        self.n_qubits = int(n_qubits)
        
        # 与 RepresentationWrapper.graph scheme 对齐的参数
        self.graph_K = 8  # gate type one-hot 维度 (H,X,Y,Z,T,CNOT,START,END)
        self.graph_F = self.graph_K + 2 * self.n_qubits  # 节点特征维度
        self.graph_Vmax = self.max_gates + 2  # 最大节点数（含 START/END）
        self.graph_state_dim = (
            self.graph_Vmax * self.graph_F +     # X_pad flattened
            self.graph_Vmax * self.graph_Vmax +  # A_pad flattened
            self.graph_Vmax                       # node_mask
        )
        
        assert self.readout in ("mean", "sum"), f"readout must be 'mean' or 'sum', got {readout}"
        
        # 输入投影层：使用 LazyLinear 自动推断输入维度
        self.in_proj = nn.LazyLinear(hid)
        
        # GIN 层
        self.gin_layers = nn.ModuleList()
        for i in range(depth):
            in_d = hid  # 第一层之后都是 hid -> hid
            out_d = hid
            self.gin_layers.append(
                GINLayer(
                    in_dim=in_d,
                    out_dim=out_d,
                    mlp_depth=mlp_depth,
                    mlp_hid=mlp_hid if mlp_hid is not None else hid,
                    eps_learnable=eps_learnable,
                    norm=norm,
                    dropout=dropout,
                    act=act,  # 传递激活函数配置
                )
            )
        
        # 输出投影 + Post processing（可选）
        # use_post_block=False 时跳过 out_proj 和 post，直接输出 hid 维度
        # 这样可以让 SharedMLP 负责后续处理，避免冗余
        if use_post_block:
            self.out_dim = int(out_dim)
            
            # 输出投影：hid -> out_dim
            if hid != out_dim:
                self.out_proj = nn.Linear(hid, out_dim)
            else:
                self.out_proj = nn.Identity()
            
            # Post processing block (与 Encoder_RNN 对齐)
            # 顺序：LayerNorm -> activation -> Dropout
            post_act_lower = (post_act or "none").lower()
            has_post = post_ln or (post_act_lower != "none") or (post_dropout > 0)
            
            if has_post:
                post_layers = []
                if post_ln:
                    post_layers.append(nn.LayerNorm(out_dim))
                if post_act_lower != "none":
                    post_layers.append(_make_activation(post_act))
                if post_dropout > 0:
                    post_layers.append(nn.Dropout(p=post_dropout))
                self.post = nn.Sequential(*post_layers)
            else:
                self.post = None
        else:
            # 不使用 post block，直接输出 hid 维度
            self.out_dim = int(hid)
            self.out_proj = None
            self.post = None

    @property
    def num_params(self) -> int:
        """
        返回可训练参数总数。
        
        注意：由于使用了 LazyLinear，需要先调用一次 forward 初始化模型。
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
                "GIN encoder has uninitialized LazyLinear. "
                "The returned count only includes initialized parameters. "
                "Call forward() with a dummy batch to initialize."
            )
        
        return total

    @classmethod
    def from_cfg(cls, *, actions, max_gates: int, cfg: dict):
        """
        工厂方法，接口与 Encoder_MLP.from_cfg 完全对齐。
        
        cfg 字段说明：
        - out_dim: encoder 输出维度（默认 256，仅 use_post_block=True 时生效）
        - hid: GIN 的 node hidden dim（默认 128）
        - depth: GIN 层数（默认 3）
        - mlp_depth: 每层 GIN 内部 MLP 的深度（默认 2）
        - mlp_hid: 每层 MLP 的隐藏维度（默认 = hid）
        - eps_learnable: GIN 的 epsilon 是否可学习（默认 True）
        - readout: "mean" / "sum" 图级 pooling（默认 "mean"）
        - use_undirected: 是否把 A 变成 A + A^T（默认 True）
        - add_self_loops: 是否给 A 加单位阵（默认 True）
        - norm: "ln" / "bn" / "none"（默认 "ln"）
        - act: GIN 内部 MLP 激活函数（默认 "relu"，支持 "silu"/"gelu" 等）
        - dropout: 默认 0.0
        - use_post_block: 是否使用 out_proj + post（默认 True）
            - True: 使用 out_proj 投影到 out_dim，并应用 post 处理，out_dim = out_dim
            - False: 跳过 out_proj 和 post，直接输出 pooled 特征，out_dim = hid
                     适合让 SharedMLP 负责后续处理，避免冗余
        - post_ln: 是否在输出后加 LayerNorm（默认 False，仅 use_post_block=True 时生效）
        - post_act: 输出后的激活函数（默认 "relu"，可设 "none" 禁用）
        - post_dropout: 输出后的 dropout（默认 0.0）
        - n_qubits: 量子比特数（用于解包 flat state vector，如未指定则从 actions 推断）
        """
        cfg = dict(cfg or {})
        
        hid = int(cfg.get("hid", 128))
        mlp_hid = cfg.get("mlp_hid", None)
        if mlp_hid is not None:
            mlp_hid = int(mlp_hid)
        
        n_qubits = cfg.get("n_qubits", None)
        if n_qubits is not None:
            n_qubits = int(n_qubits)
        
        return cls(
            actions=actions,
            max_gates=max_gates,
            out_dim=int(cfg.get("out_dim", 256)),
            hid=hid,
            depth=int(cfg.get("depth", 3)),
            mlp_depth=int(cfg.get("mlp_depth", 2)),
            mlp_hid=mlp_hid,
            eps_learnable=bool(cfg.get("eps_learnable", True)),
            readout=str(cfg.get("readout", "mean")),
            use_undirected=bool(cfg.get("use_undirected", True)),
            add_self_loops=bool(cfg.get("add_self_loops", True)),
            norm=str(cfg.get("norm", "ln")),
            act=str(cfg.get("act", "relu")),  # GIN 内部 MLP 激活函数
            dropout=float(cfg.get("dropout", 0.0)),
            use_post_block=bool(cfg.get("use_post_block", True)),
            post_ln=bool(cfg.get("post_ln", False)),
            post_act=str(cfg.get("post_act", "relu")),
            post_dropout=float(cfg.get("post_dropout", 0.0)),
            n_qubits=n_qubits,
        )
    
    def _parse_input(self, x) -> tuple:
        """
        解析输入，返回 (X, A, mask) 元组。
        
        支持格式：
        1. dict with "state" key -> 递归解析 state
        2. dict: {"x": X, "a": A, "mask": M}
        3. tuple/list: (X, A) 或 (X, A, M)
        4. flat tensor/ndarray: [B, graph_state_dim] 或 [graph_state_dim]（来自 RepresentationWrapper）
        """
        # 兼容 obs=dict 且含 "state" key 的情况
        if isinstance(x, dict) and "state" in x:
            x = x["state"]
        
        # dict 格式
        if isinstance(x, dict):
            # 使用 is not None 检查，避免 tensor 的布尔值歧义
            X = None
            for key in ("x", "X", "node_features"):
                if key in x and x[key] is not None:
                    X = x[key]
                    break
            
            A = None
            for key in ("a", "A", "adj", "adjacency"):
                if key in x and x[key] is not None:
                    A = x[key]
                    break
            
            mask = None
            for key in ("mask", "node_mask"):
                if key in x and x[key] is not None:
                    mask = x[key]
                    break
            
            if X is None or A is None:
                raise ValueError(
                    "Encoder_GIN requires graph input with 'x' (node features) and 'a' (adjacency matrix). "
                    "If you are using gate_seq scheme, please switch to a graph-based representation scheme."
                )
            
            return X, A, mask
        
        # tuple/list 格式
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                return x[0], x[1], None
            elif len(x) == 3:
                return x[0], x[1], x[2]
            else:
                raise ValueError(f"Expected tuple of length 2 or 3, got {len(x)}")
        
        # ===== 新增：flat tensor/ndarray 格式（来自 RepresentationWrapper.graph）=====
        # 检测是否是 flat vector
        is_flat = False
        if isinstance(x, np.ndarray):
            is_flat = (x.ndim == 1) or (x.ndim == 2 and x.shape[-1] == self.graph_state_dim)
        elif isinstance(x, torch.Tensor):
            is_flat = (x.dim() == 1) or (x.dim() == 2 and x.shape[-1] == self.graph_state_dim)
        
        if is_flat:
            return self._unpack_flat_state(x)
        
        # 既不是 dict 也不是 tuple/list 也不是 flat vector
        raise ValueError(
            f"Encoder_GIN requires graph input (dict with 'x'/'a', tuple (X, A), or flat state vector), "
            f"but got {type(x).__name__} with shape {getattr(x, 'shape', 'unknown')}. "
            "If you are using gate_seq scheme, please switch to a graph-based representation scheme."
        )
    
    def _unpack_flat_state(self, state_vec) -> tuple:
        """
        将 flat state vector 解包为 (X, A, node_mask)。
        与 RepresentationWrapper.unpack_graph_state() 对齐。
        
        Args:
            state_vec: shape = [B, graph_state_dim] 或 [graph_state_dim]
        
        Returns:
            X:         节点特征矩阵, shape = [B, Vmax, F] 或 [Vmax, F]
            A:         邻接矩阵, shape = [B, Vmax, Vmax] 或 [Vmax, Vmax]
            node_mask: 有效节点 mask, shape = [B, Vmax] 或 [Vmax]
        """
        Vmax = self.graph_Vmax
        F = self.graph_F
        
        # 转换为 numpy 方便切片
        is_tensor = isinstance(state_vec, torch.Tensor)
        if is_tensor:
            was_tensor = True
            device = state_vec.device
            dtype = state_vec.dtype
            state_vec = state_vec.cpu().numpy()
        else:
            was_tensor = False
        
        # 处理 batch 维度
        if state_vec.ndim == 1:
            state_vec = state_vec[np.newaxis, :]  # [1, D]
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B = state_vec.shape[0]
        
        # 计算切分点
        x_size = Vmax * F
        a_size = Vmax * Vmax
        
        # 切片并 reshape
        X = state_vec[:, :x_size].reshape(B, Vmax, F)
        A = state_vec[:, x_size:x_size + a_size].reshape(B, Vmax, Vmax)
        node_mask = state_vec[:, x_size + a_size:]
        
        # 还原 batch 维度
        if squeeze_batch:
            X = X[0]          # [Vmax, F]
            A = A[0]          # [Vmax, Vmax]
            node_mask = node_mask[0]  # [Vmax]
        
        # 还原 tensor
        if was_tensor:
            X = torch.from_numpy(X).to(device=device, dtype=dtype)
            A = torch.from_numpy(A).to(device=device, dtype=dtype)
            node_mask = torch.from_numpy(node_mask).to(device=device, dtype=dtype)
        
        return X, A, node_mask
    
    def _prepare_tensors(self, X, A, mask, device):
        """
        准备张量：转换类型、添加 batch 维度、处理邻接矩阵。
        """
        # numpy -> torch
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # 确保是 float
        X = X.float()
        A = A.float()
        
        # [N, F] -> [1, N, F]
        if X.dim() == 2:
            X = X.unsqueeze(0)
        # [N, N] -> [1, N, N]
        if A.dim() == 2:
            A = A.unsqueeze(0)
        # [N] -> [1, N]
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        B, N, F = X.shape
        
        # 如果没有 mask，默认全 1
        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=torch.float32)
        else:
            mask = mask.float()
        
        # 移动到目标设备
        X = X.to(device)
        A = A.to(device)
        mask = mask.to(device)
        
        # 处理邻接矩阵
        if self.use_undirected:
            # A -> A + A^T（对称化）
            A = A + A.transpose(-1, -2)
            # 避免重复边：clamp 到 1
            A = A.clamp(max=1.0)
        
        if self.add_self_loops:
            # 添加自环：A + I
            eye = torch.eye(N, device=device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
            A = A + eye
            A = A.clamp(max=1.0)
        
        return X, A, mask
    
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: 图输入，支持多种格式（见 _parse_input）
        
        Returns:
            [B, out_dim] 的图级特征向量
        """
        device = next(self.parameters()).device
        
        # 解析输入
        X, A, mask = self._parse_input(x)
        
        # 准备张量
        X, A, mask = self._prepare_tensors(X, A, mask, device)
        
        B, N, F = X.shape
        
        # 边界情况：空图（N=0 或 mask 全 0）
        if N == 0:
            return torch.zeros(B, self.out_dim, device=device, dtype=X.dtype)
        
        mask_sum = mask.sum(dim=1, keepdim=True)  # [B, 1]
        if (mask_sum == 0).all():
            return torch.zeros(B, self.out_dim, device=device, dtype=X.dtype)
        
        # 输入投影：[B, N, F] -> [B, N, hid]
        H = self.in_proj(X)
        
        # GIN 层
        for gin_layer in self.gin_layers:
            H = gin_layer(H, A)
        
        # Graph pooling with mask
        # mask: [B, N] -> [B, N, 1]
        mask_expanded = mask.unsqueeze(-1)
        
        if self.readout == "mean":
            # masked mean pooling
            H_masked = H * mask_expanded  # [B, N, hid]
            H_sum = H_masked.sum(dim=1)   # [B, hid]
            # 避免除以 0
            mask_sum_clamped = mask_sum.clamp(min=1.0)  # [B, 1]
            pooled = H_sum / mask_sum_clamped  # [B, hid]
        else:  # sum
            H_masked = H * mask_expanded
            pooled = H_masked.sum(dim=1)  # [B, hid]
        
        # 输出投影 + Post processing（可选）
        if self.use_post_block:
            z = self.out_proj(pooled)  # [B, out_dim]
            
            # Post processing (LayerNorm -> activation -> Dropout)
            if self.post is not None:
                z = self.post(z)
        else:
            # 不使用 post block，直接输出 pooled 特征 [B, hid]
            z = pooled
        
        # 处理完全空的样本（mask_sum = 0）
        # 必须在最后处理，因为 Linear/LayerNorm 都有可学习 bias
        empty_mask = (mask_sum.squeeze(-1) == 0)  # [B]
        if empty_mask.any():
            z = torch.where(
                empty_mask.unsqueeze(-1),
                torch.zeros_like(z),
                z
            )
        
        return z


def build_encoder_gin(*, actions, max_gates: int, net_cfg: dict | None = None) -> Encoder_GIN:
    """工厂函数：train.py 里一行就能构造。"""
    return Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg or {})


if __name__ == "__main__":
    """
    Self-test: test Encoder_GIN basic functionality
    """
    print("=" * 60)
    print("Encoder_GIN Self-Test")
    print("=" * 60)
    
    # Mock params
    actions = [f"gate_{i}" for i in range(32)]
    max_gates = 20
    
    # Create encoder (with post block enabled for testing)
    cfg = {
        "out_dim": 256,
        "hid": 64,
        "depth": 2,
        "mlp_depth": 2,
        "eps_learnable": True,
        "readout": "mean",
        "use_undirected": True,
        "add_self_loops": True,
        "norm": "ln",
        "dropout": 0.0,
        # Post block parameters (aligned with Encoder_RNN)
        "post_ln": True,
        "post_act": "relu",
        "post_dropout": 0.0,
    }
    encoder = Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg)
    print(f"\n[Config] out_dim={encoder.out_dim}, hid={encoder.hid}, depth={encoder.depth}")
    print(f"[Config] post_ln=True, post_act=relu, post_dropout=0.0")
    
    # Test 1: dict format input
    print("\n--- Test 1: dict format input ---")
    B, N, F = 4, 10, 16
    X = torch.randn(B, N, F)
    A = (torch.rand(B, N, N) > 0.8).float()
    mask = torch.ones(B, N)
    
    state = {"x": X, "a": A, "mask": mask}
    z = encoder({"state": state})
    print(f"Input: X={X.shape}, A={A.shape}, mask={mask.shape}")
    print(f"Output: z={z.shape}")
    assert z.shape == (B, 256), f"Expected ({B}, 256), got {z.shape}"
    print("[PASS] Test 1")
    
    # Test 2: tuple format input
    print("\n--- Test 2: tuple format input ---")
    z2 = encoder((X, A, mask))
    print(f"Input: tuple (X, A, mask)")
    print(f"Output: z={z2.shape}")
    assert z2.shape == (B, 256), f"Expected ({B}, 256), got {z2.shape}"
    print("[PASS] Test 2")
    
    # Test 3: mask all 0 (empty graph) with post_ln=True
    # Critical: even with LayerNorm (which has learnable bias), empty graph must output all zeros
    print("\n--- Test 3: mask all 0 (empty graph, post_ln=True) ---")
    mask_zero = torch.zeros(B, N)
    z3 = encoder({"state": {"x": X, "a": A, "mask": mask_zero}})
    print(f"Input: mask all 0 (encoder has post_ln=True)")
    print(f"Output: z={z3.shape}, all zeros: {(z3 == 0).all().item()}")
    assert z3.shape == (B, 256), f"Expected ({B}, 256), got {z3.shape}"
    assert (z3 == 0).all(), "Expected all zeros for empty graph even with post_ln=True"
    print("[PASS] Test 3 (post_ln=True, empty graph still outputs zeros)")
    
    # Test 4: no batch dim input [N, F]
    print("\n--- Test 4: no batch dim input [N, F] ---")
    X_2d = torch.randn(N, F)
    A_2d = (torch.rand(N, N) > 0.8).float()
    z4 = encoder((X_2d, A_2d))
    print(f"Input: X={X_2d.shape}, A={A_2d.shape}")
    print(f"Output: z={z4.shape}")
    assert z4.shape == (1, 256), f"Expected (1, 256), got {z4.shape}"
    print("[PASS] Test 4")
    
    # Test 5: numpy input
    print("\n--- Test 5: numpy input ---")
    X_np = np.random.randn(B, N, F).astype(np.float32)
    A_np = (np.random.rand(B, N, N) > 0.8).astype(np.float32)
    mask_np = np.ones((B, N), dtype=np.float32)
    z5 = encoder({"state": {"x": X_np, "a": A_np, "mask": mask_np}})
    print(f"Input: numpy arrays")
    print(f"Output: z={z5.shape}")
    assert z5.shape == (B, 256), f"Expected ({B}, 256), got {z5.shape}"
    print("[PASS] Test 5")
    
    # Test 6: sum readout
    print("\n--- Test 6: sum readout ---")
    cfg_sum = dict(cfg)
    cfg_sum["readout"] = "sum"
    encoder_sum = Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg_sum)
    z6 = encoder_sum({"state": {"x": X, "a": A, "mask": mask}})
    print(f"Output: z={z6.shape}")
    assert z6.shape == (B, 256), f"Expected ({B}, 256), got {z6.shape}"
    print("[PASS] Test 6")
    
    # Test 7: no post block (post_ln=False, post_act="none")
    print("\n--- Test 7: no post block ---")
    cfg_no_post = dict(cfg)
    cfg_no_post["post_ln"] = False
    cfg_no_post["post_act"] = "none"
    cfg_no_post["post_dropout"] = 0.0
    encoder_no_post = Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg_no_post)
    assert encoder_no_post.post is None, "Expected no post block when all post options disabled"
    z7 = encoder_no_post({"state": {"x": X, "a": A, "mask": mask}})
    assert z7.shape == (B, 256), f"Expected ({B}, 256), got {z7.shape}"
    # Also verify empty graph still works without post block
    z7_empty = encoder_no_post({"state": {"x": X, "a": A, "mask": mask_zero}})
    assert (z7_empty == 0).all(), "Expected all zeros for empty graph without post block"
    print(f"Output: z={z7.shape}, post=None, empty graph zeros: {(z7_empty == 0).all().item()}")
    print("[PASS] Test 7")
    
    # Test 8: partial mask (some samples empty, some not) with post_ln=True
    print("\n--- Test 8: partial mask with post_ln=True ---")
    mask_partial = torch.ones(B, N)
    mask_partial[0, :] = 0  # Sample 0 is empty
    mask_partial[2, :] = 0  # Sample 2 is empty
    z8 = encoder({"state": {"x": X, "a": A, "mask": mask_partial}})
    assert z8.shape == (B, 256), f"Expected ({B}, 256), got {z8.shape}"
    assert (z8[0] == 0).all(), "Sample 0 (empty) should be all zeros"
    assert (z8[2] == 0).all(), "Sample 2 (empty) should be all zeros"
    assert not (z8[1] == 0).all(), "Sample 1 (non-empty) should not be all zeros"
    assert not (z8[3] == 0).all(), "Sample 3 (non-empty) should not be all zeros"
    print(f"Output: z={z8.shape}, empty samples [0,2] are zeros, non-empty [1,3] are non-zero")
    print("[PASS] Test 8")
    
    # Test 9: use_post_block=False (skip out_proj + post, output hid dim)
    print("\n--- Test 9: use_post_block=False ---")
    cfg_no_postblock = {
        "hid": 64,
        "depth": 2,
        "out_dim": 256,  # This should be ignored when use_post_block=False
        "use_post_block": False,
    }
    encoder_no_postblock = Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=cfg_no_postblock)
    # out_dim should be hid (64) when use_post_block=False
    assert encoder_no_postblock.out_dim == 64, f"Expected out_dim=64 (hid), got {encoder_no_postblock.out_dim}"
    assert encoder_no_postblock.out_proj is None, "out_proj should be None when use_post_block=False"
    assert encoder_no_postblock.post is None, "post should be None when use_post_block=False"
    z9 = encoder_no_postblock({"state": {"x": X, "a": A, "mask": mask}})
    assert z9.shape == (B, 64), f"Expected ({B}, 64), got {z9.shape}"
    # Also verify empty graph works
    z9_empty = encoder_no_postblock({"state": {"x": X, "a": A, "mask": mask_zero}})
    assert (z9_empty == 0).all(), "Expected all zeros for empty graph"
    print(f"Output: z={z9.shape}, out_dim={encoder_no_postblock.out_dim} (=hid)")
    print(f"  out_proj=None, post=None, empty graph zeros: {(z9_empty == 0).all().item()}")
    print("[PASS] Test 9 (use_post_block=False, let SharedMLP handle post-processing)")
    
    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
