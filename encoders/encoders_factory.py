"""
Encoder 工厂函数：根据 net_cfg 中的 encoder 类型选择对应的 encoder 类。

用法：
    from encoders.encoders_factory import build_encoder
    
    encoder = build_encoder(
        actions=actions,
        max_gates=max_gates,
        net_cfg=net_cfg
    ).to(device)

支持的 encoder 类型（通过 net_cfg["encoder"] 或 net_cfg["enc_name"] 或 net_cfg["enc_type"] 指定）：
    - "mlp": Encoder_MLP（默认）
    - "rnn": Encoder_RNN
    - "gin": Encoder_GIN
    - "cnn": Encoder_CNN

参数量工具：
    from encoders.encoders_factory import estimate_params, suggest_config
    
    # 估算给定配置的参数量
    params = estimate_params("mlp", actions, max_gates, net_cfg)
    
    # 给定目标参数量，推荐配置
    cfg = suggest_config("rnn", actions, max_gates, target_params=100_000)
"""
import math

from .encoder_MLP import Encoder_MLP
from .encoder_RNN import Encoder_RNN
from .encoder_GIN import Encoder_GIN
from .encoder_CNN import Encoder_CNN, build_encoder_cnn


def build_encoder(*, actions, max_gates: int, net_cfg: dict):
    """
    根据 net_cfg 构建对应的 encoder。
    
    Args:
        actions: 动作列表（create_action_mapping 生成）
        max_gates: 最大门数（observation 长度）
        net_cfg: 网络配置字典，包含：
            - encoder / enc_name / enc_type: encoder 类型（默认 "mlp"）
            - 其他 encoder 特定的超参数
    
    Returns:
        encoder: nn.Module，具有 .out_dim 属性和 forward 方法
    
    Raises:
        ValueError: 未知的 encoder 类型
    
    Examples:
        # MLP encoder
        net_cfg = {"encoder": "mlp", "hid": 256, "depth": 2}
        encoder = build_encoder(actions=actions, max_gates=20, net_cfg=net_cfg)
        
        # GIN encoder
        net_cfg = {
            "encoder": "gin",
            "hid": 128,
            "depth": 3,
            "out_dim": 256,
            "mlp_depth": 2,
            "readout": "mean",
        }
        encoder = build_encoder(actions=actions, max_gates=20, net_cfg=net_cfg)
    """
    net_cfg = dict(net_cfg or {})
    
    # 获取 encoder 类型（兼容多种字段名）
    enc_name = (
        net_cfg.get("encoder") or 
        net_cfg.get("enc_name") or 
        net_cfg.get("enc_type") or 
        "mlp"
    ).lower()
    
    if enc_name == "mlp":
        return Encoder_MLP.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg)
    
    if enc_name == "rnn":
        return Encoder_RNN.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg)
    
    if enc_name == "gin":
        return Encoder_GIN.from_cfg(actions=actions, max_gates=max_gates, cfg=net_cfg)
    
    if enc_name in ("cnn", "encoder_cnn"):
        return build_encoder_cnn(actions=actions, max_gates=max_gates, net_cfg=net_cfg)
    
    raise ValueError(
        f"Unknown encoder type: '{enc_name}'. "
        f"Supported types: 'mlp', 'rnn', 'gin', 'cnn'."
    )


# ============================================================================
# 参数量估算工具
# ============================================================================

def estimate_params(
    enc_type: str,
    actions,
    max_gates: int,
    net_cfg: dict | None = None,
    *,
    # CNN/GIN 特殊参数（因为它们的输入维度不依赖 actions/max_gates）
    cnn_input_channels: int = 4,  # grid 模式默认 4 channels
    gin_node_feat_dim: int = 16,  # GIN 节点特征维度
) -> int:
    """
    估算给定 encoder 配置的参数量（不实际构建模型）。
    
    Args:
        enc_type: encoder 类型 ("mlp", "rnn", "cnn", "gin")
        actions: 动作列表（用于计算 vocab size）
        max_gates: 最大门数
        net_cfg: encoder 配置字典
        cnn_input_channels: CNN 输入通道数（grid=4, tensor 需指定）
        gin_node_feat_dim: GIN 节点特征维度
    
    Returns:
        估算的参数量
    
    Examples:
        >>> actions = [f"gate_{i}" for i in range(32)]
        >>> estimate_params("mlp", actions, 60, {"hid": 256, "depth": 2})
        575488
    """
    enc_type = enc_type.lower()
    cfg = dict(net_cfg or {})
    A = len(actions)
    
    if enc_type == "mlp":
        return _estimate_mlp_params(A, max_gates, cfg)
    elif enc_type == "rnn":
        return _estimate_rnn_params(A, cfg)
    elif enc_type in ("cnn", "encoder_cnn"):
        return _estimate_cnn_params(cnn_input_channels, cfg)
    elif enc_type == "gin":
        return _estimate_gin_params(gin_node_feat_dim, cfg)
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


def _estimate_mlp_params(A: int, max_gates: int, cfg: dict) -> int:
    """估算 MLP 参数量"""
    V = A + 1  # vocab size (含 PAD)
    in_dim = max_gates * V
    
    hid = int(cfg.get("hid", 256))
    depth = int(cfg.get("depth", 2))
    out_dim = int(cfg.get("out_dim", hid))
    hidden_dims = cfg.get("hidden_dims", None)
    use_ln = bool(cfg.get("use_ln", False))
    
    if hidden_dims is None:
        hidden_dims = [hid] * depth
    else:
        hidden_dims = [int(x) for x in hidden_dims]
    
    dims = [in_dim] + hidden_dims + [out_dim]
    
    total = 0
    for i in range(len(dims) - 1):
        # Linear: weight + bias
        total += dims[i] * dims[i + 1] + dims[i + 1]
        
        is_last = (i == len(dims) - 2)
        if not is_last and use_ln:
            # LayerNorm: gamma + beta
            total += 2 * dims[i + 1]
    
    return total


def _estimate_rnn_params(A: int, cfg: dict) -> int:
    """估算 RNN 参数量"""
    vocab_size = A + 1  # 含 PAD
    
    embed_dim = int(cfg.get("embed_dim", 64))
    pre_dim = cfg.get("pre_dim", None)
    rnn_type = str(cfg.get("rnn_type", "lstm")).lower()
    hidden_size = int(cfg.get("hidden_size", 100))
    num_layers = int(cfg.get("num_layers", 1))
    bidirectional = bool(cfg.get("bidirectional", False))
    out_dim = cfg.get("out_dim", None)
    post_ln = bool(cfg.get("post_ln", False))
    
    num_directions = 2 if bidirectional else 1
    rnn_out_dim = hidden_size * num_directions
    if out_dim is None:
        out_dim = rnn_out_dim
    else:
        out_dim = int(out_dim)
    
    total = 0
    
    # Embedding (PAD idx 不影响参数量)
    total += vocab_size * embed_dim
    
    # Pre-coding layer (可选)
    rnn_input_dim = embed_dim
    if pre_dim is not None:
        pre_dim = int(pre_dim)
        total += embed_dim * pre_dim + pre_dim  # Linear
        total += 2 * pre_dim  # LayerNorm
        rnn_input_dim = pre_dim
    
    # RNN 参数（LSTM 有 4 个 gate，GRU 有 3 个 gate）
    gate_mult = 4 if rnn_type == "lstm" else 3
    
    for layer in range(num_layers):
        if layer == 0:
            input_sz = rnn_input_dim
        else:
            input_sz = hidden_size * num_directions
        
        # 每个 direction 的参数
        # weight_ih: gate_mult * hidden_size * input_sz
        # weight_hh: gate_mult * hidden_size * hidden_size
        # bias_ih + bias_hh: 2 * gate_mult * hidden_size
        layer_params = num_directions * (
            gate_mult * hidden_size * input_sz +
            gate_mult * hidden_size * hidden_size +
            2 * gate_mult * hidden_size
        )
        total += layer_params
    
    # Projection layer (可选)
    if out_dim != rnn_out_dim:
        total += rnn_out_dim * out_dim + out_dim
    
    # Post LayerNorm (可选)
    if post_ln:
        total += 2 * out_dim
    
    return total


def _estimate_cnn_params(input_channels: int, cfg: dict) -> int:
    """
    估算 CNN 参数量。
    
    根据 mode 决定估算哪个分支：
    - mode="grid": 1D CNN (Conv1d)，kernel_size 是 1D
    - mode="tensor": 2D CNN (Conv2d)，kernel_size 是 2D (kernel_size x kernel_size)
    
    Args:
        input_channels: 输入通道数
            - grid 模式 (mode="grid"): 默认 4 (4 行作为 4 channel)
            - tensor 模式 (mode="tensor"): 通常是 7 (gate_type channels)
        cfg: encoder 配置字典，包含 mode, hid, depth, kernel_size, out_dim, use_proj 等
    """
    hid = int(cfg.get("hid", 64))
    depth = int(cfg.get("depth", 3))
    kernel_size = int(cfg.get("kernel_size", 3))
    out_dim = int(cfg.get("out_dim", 256))
    use_proj = bool(cfg.get("use_proj", True))
    mode = str(cfg.get("mode", "grid")).lower()
    
    total = 0
    
    if mode == "tensor":
        # ========== Tensor 分支 (2D CNN) ==========
        # Conv2d: kernel 是 2D (kernel_size x kernel_size)
        # 第一层 Conv2d: in_channels -> hid
        total += input_channels * hid * kernel_size * kernel_size + hid  # weight + bias
        
        # 堆叠的 Conv2d layers
        for _ in range(depth):
            total += hid * hid * kernel_size * kernel_size + hid
        
        # 投影层
        if use_proj:
            total += hid * out_dim + out_dim
    else:
        # ========== Grid 分支 (1D CNN) ==========
        # Conv1d: kernel 是 1D (kernel_size)
        # 第一层 Conv1d: in_channels -> hid
        total += input_channels * hid * kernel_size + hid  # weight + bias
        
        # 堆叠的 Conv1d layers
        for _ in range(depth):
            total += hid * hid * kernel_size + hid
        
        # 投影层
        if use_proj:
            total += hid * out_dim + out_dim
    
    return total


def _estimate_gin_params(node_feat_dim: int, cfg: dict) -> int:
    """估算 GIN 参数量"""
    hid = int(cfg.get("hid", 128))
    depth = int(cfg.get("depth", 3))
    mlp_depth = int(cfg.get("mlp_depth", 2))
    mlp_hid = cfg.get("mlp_hid", None)
    if mlp_hid is None:
        mlp_hid = hid
    else:
        mlp_hid = int(mlp_hid)
    out_dim = int(cfg.get("out_dim", 256))
    use_post_block = bool(cfg.get("use_post_block", True))
    post_ln = bool(cfg.get("post_ln", False))
    
    total = 0
    
    # 输入投影层: node_feat_dim -> hid
    total += node_feat_dim * hid + hid
    
    # GIN 层
    for i in range(depth):
        in_d = hid
        out_d = hid
        
        # GIN 内部 MLP
        dims = [in_d] + [mlp_hid] * (mlp_depth - 1) + [out_d]
        for j in range(len(dims) - 1):
            total += dims[j] * dims[j + 1] + dims[j + 1]  # Linear
            is_last = (j == len(dims) - 2)
            if not is_last:
                total += 2 * dims[j + 1]  # LayerNorm
        
        # epsilon 参数
        total += 1
    
    # 输出层
    if use_post_block:
        if hid != out_dim:
            total += hid * out_dim + out_dim  # out_proj
        if post_ln:
            total += 2 * out_dim
    
    return total


# ============================================================================
# 参数量对齐辅助函数
# ============================================================================

def suggest_config(
    enc_type: str,
    actions,
    max_gates: int,
    target_params: int,
    *,
    tolerance: float = 0.1,
    cnn_input_channels: int = 4,
    gin_node_feat_dim: int = 16,
    base_cfg: dict | None = None,
) -> dict:
    """
    给定目标参数量，推荐合适的 encoder 配置。
    
    Args:
        enc_type: encoder 类型 ("mlp", "rnn", "cnn", "gin")
        actions: 动作列表
        max_gates: 最大门数
        target_params: 目标参数量
        tolerance: 允许的偏差比例（默认 10%）
        cnn_input_channels: CNN 输入通道数
        gin_node_feat_dim: GIN 节点特征维度
        base_cfg: 基础配置（会在此基础上调整关键参数）
    
    Returns:
        推荐的配置字典，包含:
        - 调整后的超参数
        - "_estimated_params": 估算的参数量
        - "_target_params": 目标参数量
        - "_deviation": 偏差比例
    
    Examples:
        >>> actions = [f"gate_{i}" for i in range(32)]
        >>> cfg = suggest_config("mlp", actions, 60, target_params=100_000)
        >>> print(cfg["hid"], cfg["_estimated_params"])
    """
    enc_type = enc_type.lower()
    cfg = dict(base_cfg or {})
    A = len(actions)
    
    if enc_type == "mlp":
        result = _suggest_mlp_config(A, max_gates, target_params, cfg)
    elif enc_type == "rnn":
        result = _suggest_rnn_config(A, target_params, cfg)
    elif enc_type in ("cnn", "encoder_cnn"):
        result = _suggest_cnn_config(cnn_input_channels, target_params, cfg)
    elif enc_type == "gin":
        result = _suggest_gin_config(gin_node_feat_dim, target_params, cfg)
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
    
    # 添加元信息
    result["_target_params"] = target_params
    result["_deviation"] = (result["_estimated_params"] - target_params) / target_params
    
    return result


def _suggest_mlp_config(A: int, max_gates: int, target: int, base_cfg: dict) -> dict:
    """
    MLP 配置建议策略：
    - 固定 depth=2（标准配置）
    - 调整 hid 以达到目标参数量
    
    MLP 参数量公式（depth=2, use_ln=False）：
    params ≈ in_dim * hid + hid * hid + hid * out_dim + 3*hid
           = hid * (in_dim + hid + out_dim + 3)
    
    设 out_dim = hid，则：
    params ≈ hid * (in_dim + 2*hid + 3) ≈ 2*hid² + in_dim*hid
    
    解二次方程求 hid。
    """
    cfg = dict(base_cfg)
    V = A + 1
    in_dim = max_gates * V
    
    depth = int(cfg.get("depth", 2))
    use_ln = bool(cfg.get("use_ln", False))
    
    # 简化：假设 out_dim = hid，depth=2
    # params ≈ in_dim * hid + hid * hid + hid * hid + hid + hid + hid
    # params ≈ in_dim * hid + 2 * hid² + 3 * hid
    # 2 * hid² + (in_dim + 3) * hid - target = 0
    
    a = depth  # 约等于 depth 个 hid*hid
    b = in_dim + 3 + (2 * depth if use_ln else 0)  # in_dim*hid + bias + ln
    c = -target
    
    # 二次公式求解
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        hid = 64  # fallback
    else:
        hid = int((-b + math.sqrt(discriminant)) / (2 * a))
    
    # 确保 hid 在合理范围内
    hid = max(32, min(hid, 2048))
    
    cfg["hid"] = hid
    cfg["depth"] = depth
    if "out_dim" not in cfg:
        cfg["out_dim"] = hid
    
    cfg["_estimated_params"] = _estimate_mlp_params(A, max_gates, cfg)
    return cfg


def _suggest_rnn_config(A: int, target: int, base_cfg: dict) -> dict:
    """
    RNN 配置建议策略：
    - 固定 embed_dim=64, num_layers=1, bidirectional=False
    - 调整 hidden_size 以达到目标参数量
    
    RNN 参数量公式（LSTM, 单层单向）：
    params ≈ vocab * embed + 4 * hs * (embed + hs) + 8 * hs + proj
           ≈ vocab * embed + 4 * hs² + 4 * embed * hs + 8 * hs + hs * out
    """
    cfg = dict(base_cfg)
    vocab_size = A + 1
    
    embed_dim = int(cfg.get("embed_dim", 64))
    rnn_type = str(cfg.get("rnn_type", "lstm")).lower()
    num_layers = int(cfg.get("num_layers", 1))
    bidirectional = bool(cfg.get("bidirectional", False))
    
    gate_mult = 4 if rnn_type == "lstm" else 3
    num_dirs = 2 if bidirectional else 1
    
    # 减去 embedding 参数
    remaining = target - vocab_size * embed_dim
    if remaining < 0:
        remaining = target // 2
    
    # 简化：假设单层，out_dim = hidden_size * num_dirs
    # params_rnn ≈ num_dirs * (gate_mult * hs * embed + gate_mult * hs² + 2 * gate_mult * hs)
    #            = num_dirs * gate_mult * (hs * embed + hs² + 2 * hs)
    # remaining ≈ num_dirs * gate_mult * hs² + num_dirs * gate_mult * (embed + 2) * hs
    
    a = num_dirs * gate_mult * num_layers
    b = num_dirs * gate_mult * (embed_dim + 2) * num_layers
    c = -remaining
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        hs = 64
    else:
        hs = int((-b + math.sqrt(discriminant)) / (2 * a))
    
    hs = max(32, min(hs, 1024))
    
    cfg["embed_dim"] = embed_dim
    cfg["hidden_size"] = hs
    cfg["num_layers"] = num_layers
    cfg["bidirectional"] = bidirectional
    cfg["rnn_type"] = rnn_type
    if "out_dim" not in cfg:
        cfg["out_dim"] = hs * num_dirs
    
    cfg["_estimated_params"] = _estimate_rnn_params(A, cfg)
    return cfg


def _suggest_cnn_config(input_channels: int, target: int, base_cfg: dict) -> dict:
    """
    CNN 配置建议策略：
    - 固定 depth=3, kernel_size=3
    - 调整 hid 以达到目标参数量
    
    CNN 参数量公式（grid 分支，use_proj=True）：
    params ≈ in_ch * hid * k + hid + depth * (hid² * k + hid) + hid * out_dim + out_dim
           ≈ depth * k * hid² + (in_ch * k + depth + out_dim + 1) * hid + out_dim
    """
    cfg = dict(base_cfg)
    
    depth = int(cfg.get("depth", 3))
    kernel_size = int(cfg.get("kernel_size", 3))
    use_proj = bool(cfg.get("use_proj", True))
    
    # 假设 out_dim = 2 * hid（常见配置）
    # params ≈ depth * k * hid² + (in_ch * k + depth + 2*hid + 1) * hid
    #        ≈ depth * k * hid² + 2 * hid² + (in_ch * k + depth + 1) * hid
    #        = (depth * k + 2) * hid² + (in_ch * k + depth + 1) * hid
    
    a = depth * kernel_size + (2 if use_proj else 0)
    b = input_channels * kernel_size + depth + 1
    c = -target
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        hid = 64
    else:
        hid = int((-b + math.sqrt(discriminant)) / (2 * a))
    
    hid = max(16, min(hid, 512))
    
    cfg["hid"] = hid
    cfg["depth"] = depth
    cfg["kernel_size"] = kernel_size
    cfg["use_proj"] = use_proj
    if "out_dim" not in cfg:
        cfg["out_dim"] = 2 * hid if use_proj else hid
    
    cfg["_estimated_params"] = _estimate_cnn_params(input_channels, cfg)
    return cfg


def _suggest_gin_config(node_feat_dim: int, target: int, base_cfg: dict) -> dict:
    """
    GIN 配置建议策略：
    - 固定 depth=3, mlp_depth=2
    - 调整 hid 以达到目标参数量
    
    GIN 参数量公式：
    params ≈ in_proj + depth * gin_layer + out_proj
           ≈ node_dim * hid + depth * (hid² * mlp_depth + ...) + hid * out_dim
           ≈ depth * mlp_depth * hid² + (node_dim + out_dim) * hid
    """
    cfg = dict(base_cfg)
    
    depth = int(cfg.get("depth", 3))
    mlp_depth = int(cfg.get("mlp_depth", 2))
    use_post_block = bool(cfg.get("use_post_block", True))
    
    # 假设 out_dim = 2 * hid
    # params ≈ depth * mlp_depth * hid² + (node_dim + 2*hid) * hid
    #        = (depth * mlp_depth + 2) * hid² + node_dim * hid
    
    a = depth * mlp_depth + (2 if use_post_block else 0)
    b = node_feat_dim
    c = -target
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        hid = 64
    else:
        hid = int((-b + math.sqrt(discriminant)) / (2 * a))
    
    hid = max(32, min(hid, 512))
    
    cfg["hid"] = hid
    cfg["depth"] = depth
    cfg["mlp_depth"] = mlp_depth
    cfg["use_post_block"] = use_post_block
    if "out_dim" not in cfg:
        cfg["out_dim"] = 2 * hid if use_post_block else hid
    
    cfg["_estimated_params"] = _estimate_gin_params(node_feat_dim, cfg)
    return cfg


def compare_params(
    actions,
    max_gates: int,
    configs: dict[str, dict],
    *,
    cnn_input_channels: int = 4,
    gin_node_feat_dim: int = 16,
) -> dict[str, int]:
    """
    比较多个 encoder 配置的参数量。
    
    Args:
        actions: 动作列表
        max_gates: 最大门数
        configs: {encoder_type: net_cfg} 字典
        cnn_input_channels: CNN 输入通道数
        gin_node_feat_dim: GIN 节点特征维度
    
    Returns:
        {encoder_type: params} 字典
    
    Examples:
        >>> actions = [f"gate_{i}" for i in range(32)]
        >>> configs = {
        ...     "mlp": {"hid": 128, "depth": 2},
        ...     "rnn": {"hidden_size": 100, "num_layers": 1},
        ...     "cnn": {"hid": 64, "depth": 3},
        ...     "gin": {"hid": 64, "depth": 3},
        ... }
        >>> result = compare_params(actions, 60, configs)
        >>> for enc, params in result.items():
        ...     print(f"{enc}: {params:,}")
    """
    result = {}
    for enc_type, cfg in configs.items():
        result[enc_type] = estimate_params(
            enc_type, actions, max_gates, cfg,
            cnn_input_channels=cnn_input_channels,
            gin_node_feat_dim=gin_node_feat_dim,
        )
    return result


# 为了方便，也导出各个 encoder 类
__all__ = [
    "build_encoder",
    "Encoder_MLP",
    "Encoder_RNN",
    "Encoder_GIN",
    "Encoder_CNN",
    # 参数量工具
    "estimate_params",
    "suggest_config",
    "compare_params",
]
