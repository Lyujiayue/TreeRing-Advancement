import cv2
import numpy as np
import torch
from PIL import Image

def get_saliency_weight_from_tensor(img_input):
    """
    输入:
        img_input: 可以是 PIL.Image 对象，也可以是 Tensor (B, C, H, W)
    输出:
        saliency_map: 归一化的权重图 (1, 1, H, W) 范围 [0, 1]，类型为 numpy.ndarray
    """

    # 1. 统一转换为 OpenCV 可用的 Numpy 格式 (H, W, C)
    if isinstance(img_input, Image.Image):
        # 如果是 PIL Image
        img = np.array(img_input)
        # PIL 通常是 RGB，OpenCV 显著性检测建议使用 BGR 效果更佳，或保持 RGB 亦可
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif torch.is_tensor(img_input):
        # 如果是 Tensor (B, C, H, W) 且范围在 [-1, 1]
        if len(img_input.shape) == 4:
            img_input = img_input[0]

        img_np = img_input.detach().cpu().numpy().transpose(1, 2, 0)
        # 映射从 [-1, 1] 到 [0, 255]
        img = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # 其他情况强制转换
        img = np.array(img_input).astype(np.uint8)

    # 2. 使用 OpenCV 静态显著性检测（频率残留法）
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(img)

    if not success:
        # 如果失败，返回全0（即不减弱水印）
        h, w = img.shape[:2]
        return np.zeros((1, 1, h, w))

    # 3. 后处理：平滑处理，避免权重突变导致视觉伪影
    # 增加核大小可以使权重过渡更自然
    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)

    # 4. 归一化并调整维度以匹配后续计算 (B, 1, H, W)
    # 确保输出范围严格在 [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() + 1e-8)
    saliency_map = saliency_map[np.newaxis, np.newaxis, :, :]

    return saliency_map