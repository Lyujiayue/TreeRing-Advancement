# 运行身份扩展与 Global B200 小规模验证方案

状态：2026-10-10，本地准备；尚未提交、推送或同步服务器，GPU 执行未授权。
已完成基线为 `6c44c0a85f625f5e2b3bb1799162a51103df5550` 的 Global CPU gate。
本文是实现与验收准备，不批准协议 v1.1、alpha grid 或任何实验运行。

## 1. 新运行身份

新版主 runner 必须显式提供 `--run_purpose smoke|full`。命名规则为：

```text
<method>_<split>_r<replicate>_<attack>[_a<alpha-token>]_<purpose>_s<start>_e<end>_g<generation-steps>_i<inversion-steps>
```

`end` 是不包含的上界。alpha 使用 Python float 的最短往返表示，`.`→`p`、
`-`→`m`、`+`→`p`；不按固定小数位截断，相邻可表示值不会合并。
`-0.0` 与 `0.0` 统一为 `0p0`。Original 不带 alpha，Global 必须带真实 alpha。
这里区分的是 runner 实际解析的浮点值，等价的输入拼写共享同一身份。

| 用途 | 示例 |
|---|---|
| 一步单样本 Original smoke | `original_development_r1_clean_smoke_s0_e1_g1_i1` |
| 正常步数单样本 Original smoke | `original_development_r1_clean_smoke_s0_e1_g50_i50` |
| 正常步数单样本 Global，alpha=0.5 | `global_development_r1_clean_a0p5_smoke_s0_e1_g50_i50` |
| 未来完整 Original Development | `original_development_r1_clean_full_s0_e32_g50_i50` |
| 未来完整 Global Development，示意 alpha=0.5 | `global_development_r1_clean_a0p5_full_s0_e32_g50_i50` |

smoke 只允许冻结 Development split；full 必须覆盖对应 split 全范围
（Development 32、Pilot 200、Formal 1000）。full 表示运行范围，不能据此认定某个
stage gate 完成。全范围 smoke 仍然是 smoke，任何低步数运行不能当作正常实验结果。
步数必须为正整数，并由验收方案另外核对；命名本身不替代协议。

旧 `original_development_r1_clean` 永久保留为历史 Stage C 单样本、一步 smoke。
历史名称读取函数不变，旧 JSON/JSONL 不迁移、不重写；新版 runner 拒绝该旧名称。
新版 `--overwrite_output` 始终拒绝，已有或中断输出保留。命名/用途/alpha 的验证发生在
模型加载、输出初始化之前。新名字不能绕过重复运行审批；重跑身份规则仍待另行审查。

schema 仍为 `treering-experiment-v2`。新增字段：

- `summary.run_identity.run_purpose`、`run_naming_version=treering-run-name-v2`。
- 逐样本同名字段，以及 `paired_inputs` 中 base latent、watermark key、mask 的 SHA-256。
- 指纹包含 dtype、shape 和连续张量的实际原始字节，在生成前保留的共享输入上计算。

这些字段使跨方法输入可审核；它们不等价于完整模型、图像质量或论文结果证明。

## 2. 拟议 GPU 验证范围

先验收数值路径，再进行小规模 E2E；两个阶段分别需要明确执行授权。
不改模型、现有 venv、CUDA/PyTorch/driver，不安装依赖，不下载模型。

### 数值 sanity

仅现有 B200 上的张量操作，不加载 SD2.1 权重，不生成图像：

- 固定形状 `[1,4,64,64]`，`float16`（真实运行路径）和 `float32`（参照路径）。
- 固定 seed 0、1，key seed 999999，rand/circle/radius=10/channel=0/complex 注入。
- alpha=0、0.5、1；各 dtype/seed 共 12 个观测，只用于实现验证。
- 精确核对 alpha=0 的独立 clone、alpha=1 的 Original 结果和中间值舍入后的公式。
- base/key/mask 不变，CPU/CUDA RNG 状态不变，dtype/device 保持一致。
- 最终实际张量的 float64 残差与独立 L2/RMS 计算一致；全体数值有限。

alpha=0.5 仅是中间实现探针，不能称为推荐 alpha 或 Pilot 候选 grid。
ComplexHalf 的实验支持 warning 是此路径预期可能出现的警告，必须原样记录；
另外只接受已知 timm deprecation warning。其他 warning 或 traceback 会阻断验收。

### 小规模端到端 smoke

只使用冻结 Development 的 index 0 / source row 1200，四次独立主 runner 调用：
Original、Global alpha=0、Global alpha=0.5、Global alpha=1。
每次自带一个 No-Watermark 配对，共 8 次生成和 8 次反演，不保存论文结果。

| 参数 | 值 |
|---|---|
| prompt 文件 | `.local_data/prompts/splits/sdp_dev_rows_1200_1231.txt`（原始 32 行文件不截断） |
| 范围 | start=0，end=1，run_purpose=smoke |
| 生成/反演步数 | 50 / 50（正常步数管线检查） |
| image_length / num_images / guidance | 512 / 1 / 7.5 |
| gen_seed / w_seed / key ID | 0 / 999999 / `treering-rand-wseed-999999` |
| pattern / radius / channel / mask | rand / 10 / 0 / circle |
| measurement / injection | l1_complex / complex |
| attack / replicate / protocol | clean / 1 / v1 |
| rand_aug / reference_model / W&B | 0 / None / disabled |
| device / dtype | 单张 NVIDIA B200 / CUDA / torch.float16；拒绝 CPU fallback |
| Python / torch / CUDA | 现有 3.12.3 / 2.7.0a0+ecf3bae40a.nv25.02 / 12.8 |
| model | `/public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base`，离线本地读取 |

4 个模型加载串行进行，每次进程退出后释放；每次最长 1800 秒。
不据旧一步 smoke 估计精确时间或费用；授权前应确认实际资源预算。

## 3. 独立脚本与验收

脚本：`scripts/global_gpu_validation.py`。默认/`plan` 只打印准备好的参数，不调用
GPU、模型或子进程。`--help` 同样只读。服务器执行要求：

1. 新命名代码经过本地审查，用户明确批准新 commit/push；随后另外批准服务器同步。
2. 清洁的 `stage-c-record-identity` 分支、用户批准的完整新 HEAD，且其祖先包含 Global CPU 基线。
   脚本拒绝使用旧 `6c44c0a` 作为新运行的验收 HEAD；不猜测未来提交 ID。
3. 现有 treering venv、持久化模型目录、冻结 prompt 与旧 Stage C 已知哈希均匹配。
4. preflight 为 `.local_outputs` 全部既有文件生成保护清单，并哈希本地模型全部文件。
   这是本次验证的模型身份快照，不是已批准的 Formal 模型 manifest。
5. 所有新输出/报告/日志路径必须未占用；新报告及 checksum 用排他创建，旧证据不可替换。

preflight 不探测 GPU、不加载模型、不运行 tests；sanity 必须显式提供
`--execute-gpu-sanity`，run-one 必须提供 `--execute-smoke`，这些开关不能替代用户授权。
run-one 仅能运行本方案四个 smoke 标签，不提供 Development/Pilot/Formal launcher。
sanity PASS 之后才开放 smoke，且报告绑定同一 HEAD 和 baseline 哈希。

最终 `accept` 核对四组 JSON/JSONL、四份 process 报告及 console 哈希：

- summary/results 与 JSONL 完全一致，恰好一个样本，显式源行/身份/范围/参数全匹配。
- HEAD、branch、dirty=false、B200、dtype、Python/torch/CUDA、prompt hash/path/count 匹配。
- 全体数值有限，必需指标存在，duration/timestamp 有效，RMS/L2 归一化正确。
- base/key/mask 指纹跨方法完全一致，No-Watermark 距离一致。
- alpha=0 的残差为零且距离等于 No-Watermark；alpha=1 的距离/预算等于 Original。
- alpha=0.5 的实际预算严格位于两个端点之间；不要求恰为 0.5 倍，保留 float16 舍入效应。
- 四个进程成功、无 traceback、无未解释 warning，数值 sanity provenance 匹配。
- preflight 保护的全部旧文件和模型哈希仍不变；生成新输出属于允许的新增文件。

跨运行端点/No-Watermark 距离要求精确相等。本地 CPU 合成路径已验证这个契约，
真实 GPU 路径尚待验证。若出现 GPU 非确定性或新 warning，停止并保存全部原始证据，
分析原因后事先审查容差/修订方案，不根据结果临时放宽验收规则。

成功状态仅为 `GLOBAL_GPU_SMALL_SMOKE_GATE_PASS`；报告明确记录
`paper_result=false`、`full_development_gate=false`、`alpha_grid_selected=false`、
`protocol_v1_1_approved=false`。本方案没有质量指标、Saliency 或固定阈值验收。

## 4. 未来服务器批次（当前不执行）

仅在相应授权和同步完成后逐批发 2–3 项短操作，等待用户返回输出再继续。
命令中的 `APPROVED_HEAD` 必须先填写已批准的新完整提交。
为减少网页终端输入，届时设置三个变量 `B`（preflight report）、`S`（sanity report）、
`H`（approved HEAD）；会话刷新后需重新 cd、激活现有 venv 和设置变量。

- 批次一：核对同步后的 branch/HEAD/clean；运行 preflight；返回报告/sha256 输出。
- 批次二：确认单张 B200 可见；获批后运行 sanity；返回报告/原始 console 输出。
- 批次三：获批后 run-one `original` 和 `global-a0`；返回两个短状态与证据。
- 批次四：run-one `global-a05` 和 `global-a1`；返回两个短状态与证据。
- 批次五：accept（携带四个 process 报告）；校验/归档全部原始文件；双端哈希核对。

完整参数在脚本内；无需用户粘贴长 runner 命令。数值失败、超时、失败进程或已有输出
都必须停下来保留 partial JSONL/log/report；不删除、不覆盖、不自动重试。
新命名部署后仍需服务器 CPU tests，不能沿用旧 31-test gate 声称新代码通过。

## 5. 仍未完成的后续 gate

本轮不运行 32 样本 Development、200 样本 Pilot 或 Formal，不选 alpha。
完整 Development 还需 Original/Global 32 prompt 配对、Saliency、质量指标等原协议要求；
Pilot 前另行冻结候选 grid 与选择规则，Pilot 冻结最终 alpha/预算，v1.1 仍待批准。
