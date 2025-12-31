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
"""

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


# 为了方便，也导出各个 encoder 类
__all__ = [
    "build_encoder",
    "Encoder_MLP",
    "Encoder_RNN",
    "Encoder_GIN",
    "Encoder_CNN",
]
