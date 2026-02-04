# CS336 Spring 2025 Assignment 5: Alignment

本目录是 **CS336 作业 5：Alignment（对齐与推理强化学习）** 的本地实践环境，包含：

- **官方英文讲义**：`cs336_spring2025_assignment5_alignment.pdf`
- **中文版讲义**：`cs336_spring2025_assignment5_alignment_zh.md`
- **补充作业（安全 & RLHF，可选）**：`cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`
- **个人学习笔记**：`README.ipynb`

如果你只是想“把代码跑起来 + 做完作业要求的实验”，可以按本文档从上到下依次执行。

---

## 1. 目录结构速览（只列与你实现/运行最相关的部分）

- `cs336_alignment/`
  - `evaluate_math.py`：**零样本基线评估脚本**（对应讲义 §3，当前默认用 GSM8K 代替 MATH）
  - `sft_math_reasoning.py`：**SFT 实验脚本**（对应讲义 §4，含数据规模 sweep + 过滤后数据实验）
  - `sft_math_reasoning_ei.py`：**专家迭代（Expert Iteration）脚本**（对应讲义 §5）
  - `grpo_experiments.py`：**GRPO 及一系列消融/排行榜实验脚本**（对应讲义 §7–9，基于 Typer 的 CLI）
  - `sft_helper.py`：你需要实现的 **SFT & 评分相关核心函数**（`tokenize_prompt_and_output` 等）
  - `gpro_helper.py`：你需要实现的 **GRPO/策略梯度相关核心函数**（`compute_group_normalized_rewards` 等）
  - `drgrpo_grader.py`：奖励函数实现（`r1_zero_reward_fn`、`question_only_reward_fn`）
  - `log.py`：日志与生成样例记录工具（`log_generations`）
  - `prompts/`：提示模板
    - `r1_zero.prompt`：默认 R1-Zero 提示
    - `question_only.prompt`：只保留 `{question}` 的提示（用于提示消融）
- `data/`
  - `gsm8k/test.jsonl`：**评测集**（当前作为 MATH 的替代品）
  - `math12k/`
    - `sft_gpt-oss-120b.jsonl`：使用教师模型根据原 math 12k 蒸馏得到的带推理轨迹 SFT 数据
    - `sft/sft_gpt-oss-120b_filtered.jsonl`：在`sft_gpt-oss-120b.jsonl`基础上，过滤只保留“答案正确”的 SFT 数据
- `results/`：各类实验输出（评测结果、日志等）
- `wandb/`：各类实验记录

更详细的背景与代码讲解，可以参考：

- 讲义中文版：`cs336_spring2025_assignment5_alignment_zh.md`
- 分步讲解文档：`README.ipynb`

---

## 2. 环境配置（uv + Python + GPU）

### 2.1 Python 与依赖管理

- 推荐使用 **Python 3.11–3.12**（`pyproject.toml` 中已限定 `>=3.11,<3.13`）
- 本仓库使用 [`uv`](https://github.com/astral-sh/uv) 管理虚拟环境与依赖
- 依赖声明见 `pyproject.toml`，主要包括：
  - `torch`、`transformers`、`vllm`、`flash-attn`、`accelerate`
  - `math-verify`、`wandb`、`pytest` 等

### 2.2 安装步骤

如同官方作业，我们分两步安装（先跳过 `flash-attn` 再整体安装）：

```bash
uv sync --no-install-package flash-attn
uv sync
```

如果你希望显式激活虚拟环境（可选）：

```bash
source .venv/bin/activate
```

### 2.3 GPU 与模型准备

- 代码默认使用 **Qwen2.5-Math-1.5B** 作为基础模型，并假设它位于：
  - `/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B`
- 如果你的模型路径不同，请修改以下文件中的 `MODEL_PATH` / `BASE_MODEL` 等常量：
  - `cs336_alignment/evaluate_math.py`
  - `cs336_alignment/sft_math_reasoning.py`
  - `cs336_alignment/sft_math_reasoning_ei.py`
  - `cs336_alignment/grpo_experiments.py` 中的 `GRPOConfig.base_model`

建议使用带有至少 80GB 显存的 GPU（运行完整 SFT / GRPO 实验时更推荐 2*80GB 级别）。

---

## 3. 运行

本节对应讲义中的各个大题，给出“直接能跑”的脚本入口，方便你做作业与复现实验。*如果你没有那么多资源从头复现，但是又想要看下结果，我们也提供了实验记录可直接访问：*

- SFT 实验记录：https://wandb.ai/xuhu0115-sju/cs336-a5-sft-v3?nw=nwuserxuhu0115
- SFT_EI 实验记录：https://wandb.ai/xuhu0115-sju/cs336-a5-sft-ei?nw=nwuserxuhu0115
- GRPO 实验记录：https://wandb.ai/xuhu0115-sju/cs336-a5-grpo?nw=nwuserxuhu0115

所有的内容讲解和结果分析都在`README.ipynb`，不知道从哪里开始的可以直接跳到这！

### 3.1 Zero-shot 基线评估（对应讲义 §3、问题 `math_baseline`）

- 脚本：`cs336_alignment/evaluate_math.py`
- 当前默认：
  - `MODEL_PATH` 指向本地 Qwen2.5-Math-1.5B
  - `MATH_VALIDATION_PATH = "data/gsm8k/test.jsonl"`（用 GSM8K 代替 MATH）
  - 提示模板使用 `cs336_alignment/prompts/r1_zero.prompt`

运行方式：

```bash
uv run python cs336_alignment/evaluate_math.py
```

运行结束后，你会得到：

- 终端打印的整体指标（`answer_accuracy` / `format_rate` / `reward_mean` 等）
- 结果文件：`results/base/zero_shot_math_evaluation.jsonl`
  - 每一行是一条样本，包含问题、模型生成、奖励拆分等信息

这些结果可以直接用于回答讲义中关于 **(1,1)/(1,0)/(0,0)** 三类样本数量与误差来源分析的问题。

### 4.2 SFT 实验（对应讲义 §4）

- 脚本：`cs336_alignment/sft_math_reasoning.py`
- 功能：
  - 在 `data/sft/sft_gpt-oss-120b.jsonl` 上做不同数据规模的 SFT（{128, 256, 512, 1024, full}）
  - 在过滤后的 `data/sft/sft_gpt-oss-120b_filtered.jsonl` 上重复上述实验
  - 使用 `evaluate_vllm` + `r1_zero_reward_fn` 在 `data/gsm8k/test.jsonl` 上评估

运行方式（一次性跑完所有 SFT 实验）：

```bash
uv run python cs336_alignment/sft_math_reasoning.py
```

输出：

- 结果目录：`results/sft_experiments_*/*/step_*/`
  - `results.jsonl`：每次评估的详细生成与奖励
  - 通过 `wandb` 记录的训练 / 评估曲线（loss、acc、熵等）

### 4.3 专家迭代（Expert Iteration，对应讲义 §5）

- 脚本：`cs336_alignment/sft_math_reasoning_ei.py`
- 功能：
  - 在当前基础模型或 SFT 模型上生成推理轨迹
  - 过滤掉错误的轨迹，只在“专家样本”上继续训练
  - 记录每一轮 EI 后验证集上的表现与 token 熵变化

运行方式（示例，实际请参考脚本头部注释和你自己的 GPU 资源情况）：

```bash
uv run python cs336_alignment/sft_math_reasoning_ei.py
```

输出：

- 对应的 `results/ei_*` 目录
- `wandb` 中的 EI 步数 vs 验证准确率 / 熵曲线

这些结果可用于回答讲义中关于 **专家迭代提升、与纯 SFT 对比** 的问题。

### 4.4 GRPO 及消融实验（对应讲义 §6–8）

- 核心脚本：`cs336_alignment/grpo_experiments.py`
- 该脚本使用 **Typer** 提供命令行接口，包含：
  - 学习率扫描：`lr_sweep`
  - 基线比较：`baselines`
  - 长度归一化：`length_norm`
  - 分组标准差归一化：`std_norm`
  - 离策略 GRPO：`off_policy` / `off_policy_sweep`
  - 裁剪消融：`clip_ablation`
  - 提示消融：`prompt_ablation`
  - 排行榜挑战：`leaderboard`

先查看所有可用子命令：

```bash
uv run python cs336_alignment/grpo_experiments.py --help
```

示例（学习率扫描实验）：

```bash
uv run python cs336_alignment/grpo_experiments.py lr-sweep
```

每个子命令会：

- 构造对应的 `GRPOConfig`（如学习率、是否使用分组标准差、长度归一化方式等）
- 调用 `run_grpo_experiment` 执行完整的 GRPO 训练循环
- 在 `results/grpo_*` 目录下保存评测结果，并通过 `wandb` 记录训练 / 评估曲线

你可以基于这些结果回答讲义中关于：

- 学习率扫描
- 基线（无基线 vs 分组归一化）
- 长度归一化（`masked_mean` vs `masked_normalize`）
- 分组标准差归一化开关
- 在策略 vs 离策略 + 裁剪是否必要
- R1-Zero 提示 vs question-only 提示

等所有 GRPO 相关问题。

---

## 5. 写作与提交（课程要求）

若完全按 Stanford CS336 的作业要求来完成，你最终需要提交：

- **`writeup.pdf`**：包含所有书面问题的答案
  - 建议在写作中直接引用：
    - 零样本基线脚本与结果
    - SFT / EI / GRPO 实验脚本对应的曲线与定量比较
    - 你在 `demo.md` 或其他笔记中整理的观察与思考
- **`code.zip`**：包含你实现的所有代码
  - 至少包括：
    - `cs336_alignment/sft_helper.py`
    - `cs336_alignment/gpro_helper.py`
    - 你修改或新增的实验脚本（如 `evaluate_math.py`、`sft_math_reasoning*.py`、`grpo_experiments.py` 等）

如果你只是将本仓库作为 **个人研究与复现实验 playground**，仍然建议：

- 保持上述文件结构不变，方便以后对照官方讲义
- 在 `README.ipynb` 或新的 Markdown 中记录你自己额外做的实验与想法

---

## 6. 常见问题（FAQ）

- **Q：刚装好环境就 `ImportError` / `CUDA error`？**
  - 请确认：
    - `torch` 版本与你的 CUDA 驱动匹配
    - GPU 上是否有足够显存（尤其是同时跑 vLLM + HF 模型时）
    - 如在 Mac / CPU 环境下，仅能跑小批量或只做逻辑测试，vLLM 相关脚本可能无法实用地跑完。

- **Q：`results/` 目录太大怎么办？**
  - 可以定期把不需要的结果目录（如早期调试版本）打包/删除，只保留你最终用于写作和复现实验的结果。

- **Q：如果作业讲义或本地实现有出入，以哪个为准？**
  - **优先以官方讲义 + tests 为准**。
  - 本地脚本（如 `evaluate_math.py` / `sft_math_reasoning.py` / `grpo_experiments.py`）是基于讲义要求做的一个“可直接跑的实现/模板”，你完全可以根据需要修改。

如有进一步想在此基础上扩展自己的研究想法，欢迎给我们`pr`!