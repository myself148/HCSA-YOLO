# Ultralytics YOLO 🚀, AGPL-3.0 license
# 额外的数据增强方法 - 补充YOLO11未实现的数据增强

import random
import numpy as np
import cv2
from typing import Tuple, Union, Optional
from ultralytics.utils.instance import Instances


class RandomNoise:
    """
    随机噪声增强类
    
    支持多种噪声类型：
    - 高斯噪声 (Gaussian)
    - 椒盐噪声 (Salt and Pepper)
    - 泊松噪声 (Poisson)
    - 均匀噪声 (Uniform)
    
    在训练前添加噪声可以提升模型对噪声的鲁棒性，特别是在实际应用中可能遇到噪声干扰的场景。
    """
    
    def __init__(self, 
                 noise_type: str = "gaussian",
                 gaussian_std: float = 0.1,
                 salt_pepper_prob: float = 0.05,
                 uniform_range: Tuple[float, float] = (-0.1, 0.1),
                 p: float = 0.5):
        """
        初始化随机噪声增强
        
        Args:
            noise_type: 噪声类型，可选: "gaussian", "salt_pepper", "poisson", "uniform"
            gaussian_std: 高斯噪声标准差 (0-1之间)
            salt_pepper_prob: 椒盐噪声概率 (0-1之间)
            uniform_range: 均匀噪声范围
            p: 应用增强的概率
        """
        self.noise_type = noise_type
        self.gaussian_std = gaussian_std
        self.salt_pepper_prob = salt_pepper_prob
        self.uniform_range = uniform_range
        self.p = p
    
    def __call__(self, labels):
        """应用噪声增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        
        if self.noise_type == "gaussian":
            # 高斯噪声
            noise = np.random.normal(0, self.gaussian_std * 255, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        elif self.noise_type == "salt_pepper":
            # 椒盐噪声
            mask = np.random.random(img.shape[:2]) < self.salt_pepper_prob
            salt = np.random.random(img.shape[:2]) < 0.5
            img[mask & salt] = 255  # 盐噪声
            img[mask & ~salt] = 0   # 椒噪声
        
        elif self.noise_type == "poisson":
            # 泊松噪声
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(img.astype(np.float32) * vals / 255.0) / vals * 255
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif self.noise_type == "uniform":
            # 均匀噪声
            noise = np.random.uniform(
                self.uniform_range[0] * 255,
                self.uniform_range[1] * 255,
                img.shape
            ).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        labels["img"] = img
        return labels


class RandomExposure:
    """
    随机曝光增强类
    
    模拟不同光照条件下的图像，提升模型对光照变化的鲁棒性。
    这对于实际应用场景非常重要，因为真实环境中的光照条件变化很大。
    """
    
    def __init__(self, 
                 exposure_range: Tuple[float, float] = (0.5, 1.5),
                 gamma_range: Tuple[float, float] = (0.7, 1.3),
                 p: float = 0.5):
        """
        初始化随机曝光增强
        
        Args:
            exposure_range: 曝光调整范围，例如(0.5, 1.5)表示50%-150%的曝光
            gamma_range: Gamma校正范围，用于模拟不同显示设备的响应
            p: 应用增强的概率
        """
        self.exposure_range = exposure_range
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, labels):
        """应用曝光增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy().astype(np.float32)
        
        # 随机曝光调整
        exposure_factor = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
        img = img * exposure_factor
        
        # 随机Gamma校正
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        inv_gamma = 1.0 / gamma
        img = np.power(img / 255.0, inv_gamma) * 255.0
        
        # 裁剪到有效范围
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        labels["img"] = img
        return labels


class RandomShadow:
    """
    随机阴影增强类
    
    在图像上添加随机阴影，模拟真实场景中的阴影效果。
    这对于提升模型在复杂光照条件下的检测能力很有帮助。
    """
    
    def __init__(self, 
                 shadow_roi: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
                 shadow_factor: Tuple[float, float] = (0.3, 0.7),
                 p: float = 0.3):
        """
        初始化随机阴影增强
        
        Args:
            shadow_roi: 阴影区域范围 (x1, y1, x2, y2)，归一化坐标
            shadow_factor: 阴影强度范围，值越小阴影越暗
            p: 应用增强的概率
        """
        self.shadow_roi = shadow_roi
        self.shadow_factor = shadow_factor
        self.p = p
    
    def __call__(self, labels):
        """应用阴影增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        h, w = img.shape[:2]
        
        # 随机生成阴影区域
        x1 = int(np.random.uniform(self.shadow_roi[0], self.shadow_roi[2]) * w)
        y1 = int(np.random.uniform(self.shadow_roi[1], self.shadow_roi[3]) * h)
        x2 = int(np.random.uniform(x1, self.shadow_roi[2] * w))
        y2 = int(np.random.uniform(y1, self.shadow_roi[3] * h))
        
        # 创建椭圆遮罩
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, 
                   ((x1 + x2) // 2, (y1 + y2) // 2),
                   ((x2 - x1) // 2, (y2 - y1) // 2),
                   0, 0, 360, 1.0, -1)
        
        # 应用阴影
        shadow_factor = np.random.uniform(self.shadow_factor[0], self.shadow_factor[1])
        img = img.astype(np.float32)
        img[mask > 0] = img[mask > 0] * shadow_factor
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        labels["img"] = img
        return labels


class RandomFog:
    """
    随机雾化增强类
    
    模拟雾天场景，提升模型在恶劣天气条件下的检测能力。
    """
    
    def __init__(self, 
                 fog_coef: Tuple[float, float] = (0.3, 0.8),
                 alpha_coef: Tuple[float, float] = (0.5, 0.9),
                 p: float = 0.2):
        """
        初始化随机雾化增强
        
        Args:
            fog_coef: 雾浓度系数范围
            alpha_coef: 混合系数范围
            p: 应用增强的概率
        """
        self.fog_coef = fog_coef
        self.alpha_coef = alpha_coef
        self.p = p
    
    def __call__(self, labels):
        """应用雾化增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        h, w = img.shape[:2]
        
        # 生成雾效果
        fog_coef = np.random.uniform(self.fog_coef[0], self.fog_coef[1])
        alpha = np.random.uniform(self.alpha_coef[0], self.alpha_coef[1])
        
        # 创建雾层（从中心向四周扩散）
        center_x, center_y = w // 2, h // 2
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        fog_mask = 1 - (distance / max_dist) * fog_coef
        fog_mask = np.clip(fog_mask, 0, 1)
        
        # 生成白色雾层
        fog = np.ones_like(img, dtype=np.float32) * 255
        
        # 混合原图和雾层
        img = img.astype(np.float32)
        img = img * (1 - alpha * fog_mask[..., np.newaxis]) + fog * (alpha * fog_mask[..., np.newaxis])
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        labels["img"] = img
        return labels


class RandomMotionBlur:
    """
    随机运动模糊增强类
    
    模拟相机或物体运动造成的模糊效果，提升模型对模糊图像的检测能力。
    """
    
    def __init__(self, 
                 kernel_size: Tuple[int, int] = (3, 15),
                 angle_range: Tuple[float, float] = (0, 360),
                 p: float = 0.3):
        """
        初始化随机运动模糊增强
        
        Args:
            kernel_size: 模糊核大小范围
            angle_range: 运动方向角度范围（度）
            p: 应用增强的概率
        """
        self.kernel_size = kernel_size
        self.angle_range = angle_range
        self.p = p
    
    def __call__(self, labels):
        """应用运动模糊增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        
        # 随机生成模糊参数
        kernel_size = np.random.randint(self.kernel_size[0], self.kernel_size[1] + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保为奇数
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        
        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        
        # 应用模糊
        img = cv2.filter2D(img, -1, kernel)
        
        labels["img"] = img
        return labels


class RandomDefocus:
    """
    随机散焦模糊增强类
    
    模拟相机对焦不准确造成的模糊效果。
    """
    
    def __init__(self, 
                 kernel_size: Tuple[int, int] = (3, 9),
                 p: float = 0.3):
        """
        初始化随机散焦模糊增强
        
        Args:
            kernel_size: 模糊核大小范围（必须是奇数）
            p: 应用增强的概率
        """
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, labels):
        """应用散焦模糊增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        
        # 随机生成模糊核大小
        kernel_size = np.random.randint(self.kernel_size[0], self.kernel_size[1] + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保为奇数
        
        # 应用高斯模糊
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        labels["img"] = img
        return labels


class RandomColorJitter:
    """
    随机颜色抖动增强类
    
    对图像的亮度、对比度、饱和度进行随机调整，比HSV增强更灵活。
    """
    
    def __init__(self, 
                 brightness: Tuple[float, float] = (0.8, 1.2),
                 contrast: Tuple[float, float] = (0.8, 1.2),
                 saturation: Tuple[float, float] = (0.8, 1.2),
                 p: float = 0.5):
        """
        初始化随机颜色抖动增强
        
        Args:
            brightness: 亮度调整范围
            contrast: 对比度调整范围
            saturation: 饱和度调整范围
            p: 应用增强的概率
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    
    def __call__(self, labels):
        """应用颜色抖动增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy().astype(np.float32)
        
        # 随机亮度调整
        brightness_factor = np.random.uniform(self.brightness[0], self.brightness[1])
        img = img * brightness_factor
        
        # 随机对比度调整
        contrast_factor = np.random.uniform(self.contrast[0], self.contrast[1])
        mean = np.mean(img)
        img = (img - mean) * contrast_factor + mean
        
        # 随机饱和度调整（转换到HSV空间）
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation_factor = np.random.uniform(self.saturation[0], self.saturation[1])
            hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # 裁剪到有效范围
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        labels["img"] = img
        return labels


class RandomGridMask:
    """
    随机网格遮罩增强类
    
    随机遮挡图像的部分区域，强制模型学习更鲁棒的特征表示。
    类似于Cutout，但使用网格模式。
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (4, 8),
                 drop_prob: float = 0.3,
                 fill_value: int = 0,
                 p: float = 0.3):
        """
        初始化随机网格遮罩增强
        
        Args:
            grid_size: 网格大小范围（行数，列数）
            drop_prob: 每个网格单元被遮挡的概率
            fill_value: 遮挡区域的填充值
            p: 应用增强的概率
        """
        self.grid_size = grid_size
        self.drop_prob = drop_prob
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, labels):
        """应用网格遮罩增强"""
        if random.random() > self.p:
            return labels
        
        img = labels["img"].copy()
        h, w = img.shape[:2]
        
        # 随机生成网格大小
        grid_h = np.random.randint(self.grid_size[0], self.grid_size[1] + 1)
        grid_w = np.random.randint(self.grid_size[0], self.grid_size[1] + 1)
        
        # 计算每个网格单元的大小
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # 随机遮挡网格单元
        for i in range(grid_h):
            for j in range(grid_w):
                if np.random.random() < self.drop_prob:
                    y1 = i * cell_h
                    y2 = min((i + 1) * cell_h, h)
                    x1 = j * cell_w
                    x2 = min((j + 1) * cell_w, w)
                    img[y1:y2, x1:x2] = self.fill_value
        
        labels["img"] = img
        return labels


class CombinedExtraAugment:
    """
    组合增强类
    
    将多个额外增强方法组合在一起使用。
    """
    
    def __init__(self, 
                 use_noise: bool = True,
                 use_exposure: bool = True,
                 use_shadow: bool = False,
                 use_fog: bool = False,
                 use_motion_blur: bool = False,
                 use_defocus: bool = False,
                 use_color_jitter: bool = True,
                 use_grid_mask: bool = False,
                 **kwargs):
        """
        初始化组合增强
        
        Args:
            use_*: 是否使用对应的增强方法
            **kwargs: 传递给各个增强方法的参数
        """
        self.augments = []
        
        if use_noise:
            noise_type = kwargs.get("noise_type", "gaussian")
            self.augments.append(RandomNoise(
                noise_type=noise_type,
                gaussian_std=kwargs.get("gaussian_std", 0.1),
                salt_pepper_prob=kwargs.get("salt_pepper_prob", 0.05),
                p=kwargs.get("noise_p", 0.3)
            ))
        
        if use_exposure:
            self.augments.append(RandomExposure(
                exposure_range=kwargs.get("exposure_range", (0.5, 1.5)),
                gamma_range=kwargs.get("gamma_range", (0.7, 1.3)),
                p=kwargs.get("exposure_p", 0.3)
            ))
        
        if use_shadow:
            self.augments.append(RandomShadow(
                shadow_roi=kwargs.get("shadow_roi", (0.0, 0.0, 1.0, 1.0)),
                shadow_factor=kwargs.get("shadow_factor", (0.3, 0.7)),
                p=kwargs.get("shadow_p", 0.2)
            ))
        
        if use_fog:
            self.augments.append(RandomFog(
                fog_coef=kwargs.get("fog_coef", (0.3, 0.8)),
                alpha_coef=kwargs.get("alpha_coef", (0.5, 0.9)),
                p=kwargs.get("fog_p", 0.15)
            ))
        
        if use_motion_blur:
            self.augments.append(RandomMotionBlur(
                kernel_size=kwargs.get("kernel_size", (3, 15)),
                angle_range=kwargs.get("angle_range", (0, 360)),
                p=kwargs.get("motion_blur_p", 0.2)
            ))
        
        if use_defocus:
            self.augments.append(RandomDefocus(
                kernel_size=kwargs.get("defocus_kernel_size", (3, 9)),
                p=kwargs.get("defocus_p", 0.2)
            ))
        
        if use_color_jitter:
            self.augments.append(RandomColorJitter(
                brightness=kwargs.get("brightness", (0.8, 1.2)),
                contrast=kwargs.get("contrast", (0.8, 1.2)),
                saturation=kwargs.get("saturation", (0.8, 1.2)),
                p=kwargs.get("color_jitter_p", 0.3)
            ))
        
        if use_grid_mask:
            self.augments.append(RandomGridMask(
                grid_size=kwargs.get("grid_size", (4, 8)),
                drop_prob=kwargs.get("drop_prob", 0.3),
                p=kwargs.get("grid_mask_p", 0.2)
            ))
    
    def __call__(self, labels):
        """应用所有增强"""
        for augment in self.augments:
            labels = augment(labels)
        return labels


class ConfigurableExtraAugment:
    """
    可通过配置文件控制的增强类
    
    这个类允许通过配置字典来控制增强方法，更适合从配置文件或命令行参数中读取配置。
    """
    
    def __init__(self, config=None):
        """
        初始化可配置增强
        
        Args:
            config: 配置字典，例如：
                {
                    "noise": {"enable": True, "type": "gaussian", "std": 0.1, "p": 0.3},
                    "exposure": {"enable": True, "range": [0.5, 1.5], "p": 0.3},
                    "shadow": {"enable": False},
                    "fog": {"enable": False},
                    "motion_blur": {"enable": False},
                    "defocus": {"enable": False},
                    "color_jitter": {"enable": True, "brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.8, 1.2], "p": 0.3},
                    "grid_mask": {"enable": False}
                }
        """
        if config is None:
            config = {}
        
        # 构建kwargs字典
        kwargs = {}
        
        # 噪声增强配置
        if config.get("noise", {}).get("enable", False):
            kwargs["use_noise"] = True
            kwargs["noise_type"] = config.get("noise", {}).get("type", "gaussian")
            kwargs["gaussian_std"] = config.get("noise", {}).get("std", 0.1)
            kwargs["salt_pepper_prob"] = config.get("noise", {}).get("salt_pepper_prob", 0.05)
            kwargs["noise_p"] = config.get("noise", {}).get("p", 0.3)
        else:
            kwargs["use_noise"] = False
        
        # 曝光增强配置
        if config.get("exposure", {}).get("enable", False):
            kwargs["use_exposure"] = True
            exposure_range = config.get("exposure", {}).get("range", [0.5, 1.5])
            kwargs["exposure_range"] = tuple(exposure_range) if isinstance(exposure_range, list) else exposure_range
            gamma_range = config.get("exposure", {}).get("gamma_range", [0.7, 1.3])
            kwargs["gamma_range"] = tuple(gamma_range) if isinstance(gamma_range, list) else gamma_range
            kwargs["exposure_p"] = config.get("exposure", {}).get("p", 0.3)
        else:
            kwargs["use_exposure"] = False
        
        # 阴影增强配置
        if config.get("shadow", {}).get("enable", False):
            kwargs["use_shadow"] = True
            shadow_roi = config.get("shadow", {}).get("roi", [0.0, 0.0, 1.0, 1.0])
            kwargs["shadow_roi"] = tuple(shadow_roi) if isinstance(shadow_roi, list) else shadow_roi
            shadow_factor = config.get("shadow", {}).get("factor", [0.3, 0.7])
            kwargs["shadow_factor"] = tuple(shadow_factor) if isinstance(shadow_factor, list) else shadow_factor
            kwargs["shadow_p"] = config.get("shadow", {}).get("p", 0.2)
        else:
            kwargs["use_shadow"] = False
        
        # 雾化增强配置
        if config.get("fog", {}).get("enable", False):
            kwargs["use_fog"] = True
            fog_coef = config.get("fog", {}).get("coef", [0.3, 0.8])
            kwargs["fog_coef"] = tuple(fog_coef) if isinstance(fog_coef, list) else fog_coef
            alpha_coef = config.get("fog", {}).get("alpha_coef", [0.5, 0.9])
            kwargs["alpha_coef"] = tuple(alpha_coef) if isinstance(alpha_coef, list) else alpha_coef
            kwargs["fog_p"] = config.get("fog", {}).get("p", 0.15)
        else:
            kwargs["use_fog"] = False
        
        # 运动模糊配置
        if config.get("motion_blur", {}).get("enable", False):
            kwargs["use_motion_blur"] = True
            kernel_size = config.get("motion_blur", {}).get("kernel_size", [3, 15])
            kwargs["kernel_size"] = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
            angle_range = config.get("motion_blur", {}).get("angle_range", [0, 360])
            kwargs["angle_range"] = tuple(angle_range) if isinstance(angle_range, list) else angle_range
            kwargs["motion_blur_p"] = config.get("motion_blur", {}).get("p", 0.2)
        else:
            kwargs["use_motion_blur"] = False
        
        # 散焦模糊配置
        if config.get("defocus", {}).get("enable", False):
            kwargs["use_defocus"] = True
            defocus_kernel_size = config.get("defocus", {}).get("kernel_size", [3, 9])
            kwargs["defocus_kernel_size"] = tuple(defocus_kernel_size) if isinstance(defocus_kernel_size, list) else defocus_kernel_size
            kwargs["defocus_p"] = config.get("defocus", {}).get("p", 0.2)
        else:
            kwargs["use_defocus"] = False
        
        # 颜色抖动配置
        if config.get("color_jitter", {}).get("enable", False):
            kwargs["use_color_jitter"] = True
            brightness = config.get("color_jitter", {}).get("brightness", [0.8, 1.2])
            kwargs["brightness"] = tuple(brightness) if isinstance(brightness, list) else brightness
            contrast = config.get("color_jitter", {}).get("contrast", [0.8, 1.2])
            kwargs["contrast"] = tuple(contrast) if isinstance(contrast, list) else contrast
            saturation = config.get("color_jitter", {}).get("saturation", [0.8, 1.2])
            kwargs["saturation"] = tuple(saturation) if isinstance(saturation, list) else saturation
            kwargs["color_jitter_p"] = config.get("color_jitter", {}).get("p", 0.3)
        else:
            kwargs["use_color_jitter"] = False
        
        # 网格遮罩配置
        if config.get("grid_mask", {}).get("enable", False):
            kwargs["use_grid_mask"] = True
            grid_size = config.get("grid_mask", {}).get("grid_size", [4, 8])
            kwargs["grid_size"] = tuple(grid_size) if isinstance(grid_size, list) else grid_size
            kwargs["drop_prob"] = config.get("grid_mask", {}).get("drop_prob", 0.3)
            kwargs["grid_mask_p"] = config.get("grid_mask", {}).get("p", 0.2)
        else:
            kwargs["use_grid_mask"] = False
        
        # 创建组合增强对象
        self.augment = CombinedExtraAugment(**kwargs)
    
    def __call__(self, labels):
        """应用增强"""
        return self.augment(labels)

