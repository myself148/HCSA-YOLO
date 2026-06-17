# Ultralytics YOLO 🚀, AGPL-3.0 license
# 数据增强集成示例
# 
# 这个文件展示了如何将额外数据增强集成到YOLO训练流程中
# 使用方法：
# 1. 复制这个文件中的代码到相应的位置
# 2. 或者直接导入并使用

from .extra_augment import CombinedExtraAugment, ConfigurableExtraAugment


def add_extra_augments_to_transforms(transforms, config=None):
    """
    将额外增强添加到现有的transforms中
    
    Args:
        transforms: Compose对象或增强列表
        config: 增强配置字典，如果为None则使用默认配置
    
    Returns:
        添加了额外增强的transforms
    """
    from .augment import Compose
    
    # 默认配置（轻量级增强）
    if config is None:
        extra_augment = CombinedExtraAugment(
            use_noise=True,
            use_exposure=True,
            use_color_jitter=True,
            noise_p=0.3,
            exposure_p=0.3,
            color_jitter_p=0.3
        )
    else:
        extra_augment = ConfigurableExtraAugment(config)
    
    # 如果transforms是Compose对象，添加增强
    if isinstance(transforms, Compose):
        transforms.append(extra_augment)
    elif isinstance(transforms, list):
        transforms.append(extra_augment)
    
    return transforms


# 使用示例配置
EXAMPLE_CONFIGS = {
    "light": {
        "noise": {"enable": True, "type": "gaussian", "std": 0.1, "p": 0.3},
        "exposure": {"enable": True, "range": [0.5, 1.5], "p": 0.3},
        "color_jitter": {"enable": True, "brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.8, 1.2], "p": 0.3}
    },
    "medium": {
        "noise": {"enable": True, "type": "gaussian", "std": 0.12, "p": 0.4},
        "exposure": {"enable": True, "range": [0.5, 1.5], "p": 0.4},
        "shadow": {"enable": True, "factor": [0.3, 0.7], "p": 0.2},
        "color_jitter": {"enable": True, "brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.8, 1.2], "p": 0.3}
    },
    "heavy": {
        "noise": {"enable": True, "type": "gaussian", "std": 0.1, "p": 0.3},
        "exposure": {"enable": True, "range": [0.5, 1.5], "p": 0.3},
        "shadow": {"enable": True, "factor": [0.3, 0.7], "p": 0.15},
        "fog": {"enable": True, "coef": [0.3, 0.8], "p": 0.1},
        "motion_blur": {"enable": True, "kernel_size": [3, 15], "p": 0.15},
        "defocus": {"enable": True, "kernel_size": [3, 9], "p": 0.15},
        "color_jitter": {"enable": True, "brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.8, 1.2], "p": 0.3}
    }
}


def get_augment_config(preset="light"):
    """
    获取预设的增强配置
    
    Args:
        preset: 预设名称，"light", "medium", 或 "heavy"
    
    Returns:
        配置字典
    """
    return EXAMPLE_CONFIGS.get(preset, EXAMPLE_CONFIGS["light"])

