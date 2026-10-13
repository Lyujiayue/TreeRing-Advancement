# Original / Global 完整 Development 准备

日期：2026-10-13。未提交、未推送、未部署、未启动实验。
本工具只准备 Original / Global 的 32 样本实现验收；完整 Development 仍需 Saliency 和质量指标。

## 修改与范围

- 主 runner 的 `--save_paired_images` 在攻击前保存配对 RGB PNG，并逐样本记录文件和像素哈希、尺寸、编码契约。
- 图像保存不增加 RNG 抽样，不改变模型或检测算法。目录/文件排他创建，失败保留原始文件。
- `--require_cuda` 在模型加载前拒绝 CPU fallback。
- `clip_similarity_available=false` 明确表示 CLIP 尚未计算；零占位值不能作为质量结果。
- `scripts/development_validation.py` 默认仅打印方案。它的执行模式和开关不能代替用户授权。
- `cpu-gate` 隐藏 GPU、离线核对实际环境，要求每个测试通过且无 skip，测试清单绑定已审查的 LF 归一化源码哈希。
- `preflight` 绑定新 HEAD、CPU gate 和原始 reviewed smoke，核对旧证据/模型/冻结输入，并冻结四组命令及每组时限。
- `run-one` 每次仅执行一组，保留 console、失败/超时状态与 partial 输出。执行后复查旧证据和模型。
- `accept` 核对四组全部 32 行、264 个文件（8 个 JSON/JSONL + 256 个 PNG）、跨方法配对与 alpha 端点，并从逐样本分数重算描述性检测指标。

## 拟议运行矩阵（须另获执行批准）

冻结 Development 的全部 32 条，gen_seed=0（逐样本 0–31）、key seed=999999，原 SD2.1、50/50 步、512、guidance=7.5、单 B200 float16、clean。
Original + Global alpha=0/0.5/1，各完整一组；每组各自生成配对 No-Watermark。
合计 128 个样本记录、256 次生成与反演。alpha 是实现探针，不选 Pilot grid 或最终参数。
每组时限必须在执行前显式批准（脚本允许 1–14400 秒）；尚未批准时间/费用预算。

运行名沿用 v2 的 `full_s0_e32_g50_i50` 规则，Global 带实际 alpha。
若输出已占用则停止；本工具不提供自动重跑、恢复或覆盖功能。必要重跑须另行审查新身份规则。

## 通过标准与后续

32 行完整、来源/seed/参数正确；共享 base/key/mask 指纹和 No-Watermark 图像一致；端点图像、检测和预算符合精确比较契约。
所有输入、代码、环境、进程/console、输出和图像哈希可复核，无未知警告、回退、遗漏或非有限数值。
原 reviewed smoke 保持原始 HEAD，不因新代码提交而改写。只接受已审查的三条完整 warning 文本。

成功只标记 `ORIGINAL_GLOBAL_32_DEVELOPMENT_GATE_PASS`；`full_development_gate=false`、`quality_metrics_validated=false`、`saliency_validated=false`。
这些图像用于随后离线质量评价；CLIP/LPIPS/PSNR/SSIM、Saliency、阈值和 v1.1 仍未完成。

## 后续用户批次

1. 审查本地改动和测试结果，确认新 commit/push。
2. 再准备可核验的离线部署包；经授权同步服务器，使用现有环境完成 CPU gate。每批给 2–3 项短操作。
3. 确认 GPU 执行预算后做 preflight；另获明确授权再逐组运行。
4. 四组输出独立验收、原始证据双端归档后继续 Saliency 与质量评价。

当前无需修改 GPU 设置、安装依赖、下载模型或重跑旧 smoke。
