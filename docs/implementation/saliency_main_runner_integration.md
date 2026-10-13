# Saliency 主 runner 集成准备

本阶段把仓库中已有的 Saliency 算法接入 Stable Diffusion 主 runner，供冻结的 Development split 做端到端实现检查。它不选择 Pilot 参数、不批准协议 v1.1，也不产生论文结果。

## 执行顺序

每条 prompt 继续先从共享 base latent 生成配对 No-Watermark 图像。Saliency 方法随后从同一 base latent 和同一 prompt 生成一张显式步数的短 DDIM preview，用仓库现有 OpenCV spectral-residual 实现构造归一化显著图，再执行已有的空间残差调制与频域重投影，最后生成水印图像并按原路径反演检测。

preview 在独立的 Python、NumPy、PyTorch CPU/CUDA 随机状态作用域内运行。退出作用域后恢复全部随机状态，因此它不会改变正式水印生成的随机输入。传入 pipeline 的 latent 也是独立 clone。

## 显式参数和身份

Saliency 必须同时显式提供：

- `--saliency_preview_steps`
- `--saliency_background_strength`
- `--saliency_foreground_strength`
- `--saliency_power`

缺失、非有限或越界参数在模型加载和输出创建前拒绝。Original 和 Global 拒绝这些参数；Saliency 拒绝 `--global_alpha`。

四个值全部进入 `method_parameters` 和 run name。示例实现探针 `preview=10, background=1.5, foreground=0.2, power=2.0` 的名称片段是：

```text
_pv10_bg1p5_fg0p2_pw2p0
```

这些值来自仓库既有 README/ImageNet runner，只用于检查实现链路，不能称为候选 grid、推荐参数或最终参数。

## 记录与验收边界

逐样本记录新增独立 `saliency` 对象，保存 map 的精确 tensor 指纹、形状、最小值、最大值、均值和 `short_ddim_preview` 来源。共享 `paired_inputs` 字段保持原有 base latent、key 和 mask 三项，便于跨方法配对核对。

合成 CPU 测试覆盖：显式参数校验、run name 身份、preview 使用共享 latent、preview 不改变正式生成 RNG、Saliency dispatch、非有限 map 拒绝、方法参数和 map 元数据写入。真实服务器 CPU gate 仍需在新 commit 部署前后执行；真实 GPU smoke 需要用户另行明确批准。

本阶段不实现或冻结 CLIP、LPIPS、PSNR、SSIM/FID，不选择 Saliency 参数，不运行 Pilot/Formal，也不把 Saliency 单方法通过解释为完整 Development gate。
