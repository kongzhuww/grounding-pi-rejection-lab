# GroundingPI 目标拒识实验室 (GroundingPI Rejection Lab)

<p align="left">
  <a href="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml"><img src="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml/badge.svg" alt="CI 状态"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

[English](README.md) | **简体中文**

**独立非官方实验性研究仓库。目前处于实验准备阶段；尚未提供训练好的模型或经过完整验证的微调后端。**

研究目标：在不牺牲真实目标定位精度的前提下，提升模型在**目标缺失/不存在（missing-target）**场景下的拒识（Rejection）能力。本实验与可运行的 [MLX 原生基线项目 (mlx-grounding-pi)](https://github.com/kongzhuww/mlx-grounding-pi) 保持相互独立，以确保基线的可复现性。初始基线 commit 已固定在 `baseline.json` 中。

## 为什么需要单独设立该实验室？

在基准开发场景中，官方原始的 CUDA BF16 模型与 Mac 8-bit 版本在面对画面中**并不存在**的黄色三角形时，均输出了具体的预测坐标而非表示不存在的 `None`。这是一个已被复现的假阳性（误报）现象，但这并不意味着模型整体表现差。本实验室旨在深入探究并改进这一行为，同时避免对基础基线代码造成静默改动。

## 当前已实现的功能

- **确定性合成场景生成**：支持针对关键点定位（Point）与检测框（BBox）任务生成包含成对“存在/缺失”属性的合成图像，输出符合原生 GAM 协议的答案字符串。
- **按场景隔离的数据切分（80/10/10）**：同一张图像/场景下的所有提问严格归入同一数据集划分，杜绝跨划分数据泄露。
- **数据防泄露校验机制**：内置图像哈希与校验工具，严格防止训练集、验证集与测试集之间的场景交叉污染。
- **全面的评估指标体系**：涵盖误报率（False Positives）、错误拒识率（False Rejection）、正例定位精度、无效格式输出率以及包含非法响应在内的负例失效度量。
- **支持本地 MLX 基线免云端评测**：可通过受信任的本地基准仓库进行评测，模型单次加载，完全无需远程推理或云端算力。
- **完善的单元测试覆盖**：16 项数据协议与退化预测器单测全部通过。
- **已完成[未经训练的本地验证冒烟测试](docs/initial-smoke.md)**：针对同一场景的两个正向问题均正常通过，两个负向问题（缺失目标）均未通过。该测试用于证明评测管线完备性，并非微调效果提升或准确率指标。

**尚未实现的内容：** 优化器/训练循环、LoRA 挂载、梯度流检查、Adapter 导出/合并或微调指标改进。`configs/lora-plan.json` 仅作为技术方案提案，非可直接执行的训练配置。不可假设纯文本的 `mlx_lm` 训练命令能直接适配此专有视觉多模态架构。

## 本地数据准备

运行环境要求 Python 3.11+。在您的 Python 环境中安装轻量级数据工具依赖：

```sh
python -m pip install -e '.[dev]'
python -m rejection_lab.data --output data/synthetic-v0 --scenes 100 --seed 42
python -m rejection_lab.data --output data/synthetic-v0 --validate-only
python -m pytest -q
```

每个合成场景包含两个带颜色的几何图形与四个测试查询：存在的目标（点/框）以及缺失的目标（点/框）。缺失查询包含颜色缺失、形状缺失以及各属性虽存在但组合不存在等情况。关键点锚点设定为圆形/方形中心点及三角形质心。坐标依据原图像素尺寸归一化量化为 0–999。该数据集属于**管线冒烟测试**，不代表在自然图像上的泛化能力；在宣称模型泛化质量前需补充独立标注的自然真实照片。已存在的输出目录不会被覆盖；若生成中断请使用新目录，验证工具会严格拦截不完整的数据集划分。生成的数据不会被 Git 追踪。

如需执行微型端到端冒烟测试，指定 `--scenes 10` 可生成 40 条数据（32 条训练、4 条验证、4 条测试）。严禁在验证集上训练或反复在测试集上调优。

## 本地基线评估（可选）

请在已安装基线 MLX 依赖的 Apple Silicon 环境中执行。该命令既不会安装新模型，也不会连接任何托管推理服务：

```sh
# 在本仓库目录下运行，指定受信任的基线环境 Python 路径：
BASELINE_REPO=/path/to/mlx-grounding-pi
"$BASELINE_REPO/.venv/bin/python" -m rejection_lab.baseline \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-8bit" \
  --dataset data/synthetic-v0 --split validation --output outputs/baseline-validation
```

基线仓库必须处于干净未修改状态。评测报告将记录其真实的 Git commit 及模型转换元数据，便于与 `baseline.json` 比对。评测使用模型常规的图像预算与贪心解码。严格的归一化适配器要求返回单一匹配标签且至多一个坐标元组；错误标签、多目标、畸变或截断输出均被计为非法输出，绝不会被误算作拒识成功。

评测指标按行消费预测结果 JSON 数据：

```json
{"id":"scene-000008-absent-point","valid":true,"coordinates":null}
```

其中 `null` 代表模型给出了明确且合规的“无目标（`None`）”判定。格式非法输出必须标记为 `valid: false`，绝不可转换为正确拒识。坐标为 0–999 的整数（点定位为两个数值，框定位为四个有序数值）。

```sh
python -m rejection_lab.metrics \
  --truth data/synthetic-v0/validation.jsonl \
  --predictions outputs/baseline-validation/predictions.jsonl \
  --output outputs/rescored-validation.json
```

若预测结果存在 ID 缺失、重复或多余，将直接报错而非从分母中静默忽略。默认阈值：在生成的 320×256 分辨率下，点误差容限为 15 像素，检测框 IoU >= 0.70。这些阈值是开发阶段的设定，非官方基准指标。恒定预测为 `None` 的退化模型虽然能获得 100% 拒识率，但正例定位得分为 0；纯非法输出同样无法通过负例失效评测。

## 启动训练前的后续步骤

1. 在场景严格解耦的正例/负例样本上确立基线基准。
2. 引入经过人工核验、版权清晰的真实自然摄影照片以及难度更高的反例样本。
3. 验证原版多模态训练路径以及冻结视觉/投影器下的 LoRA 梯度流动；细致检查 Token 损失掩码（Loss Mask）与实际可训练参数分布。
4. 开展微量过拟合与权重重载校验，随后进行受控验证集效果比对。
5. 仅在架构完全兼容且算力预算确认的前提下，再扩大训练规模并在锁定测试集上进行终测。

详见[实验协议与计划说明](docs/experiment-plan.md)。目前尚未开展任何付费云端训练。本仓库不包含任何私有凭据、模型权重文件、私人照片或生成的中间数据集。

## 开源许可证

实验室原创代码基于 Apache-2.0 许可证开源。衍生自 GroundingPI / Kimi 的代码、原始权重及后续衍生的 Adapter 均保留原有的上游许可条款；本仓库不擅自修改其许可属性。详见 [第三方开源声明](THIRD_PARTY_NOTICES.md) 与保留的许可证正文。
