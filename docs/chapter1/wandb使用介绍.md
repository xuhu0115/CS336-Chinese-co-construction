# Weights & Biases（W&B）使用介绍

`Weights & Biases（W&B）`是一个专为机器学习实验设计的协作平台，支持实验配置（config）自动记录、指标（metrics）实时可视化、超参数搜索（Sweeps）、模型版本管理、代码快照与数据集追踪和多人协作与报告生成等内容。在大模型研究中，W&B 能显著提升实验的可追溯性、可复现性与分析效率。

如果你是第一次使用它，请先参考官方文档[创建账号](https://wandb.ai/login?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=quickstart)并设置[API key](https://wandb.ai/settings)。接下里我会带你快速入门此工具。

## 1. 安装与登录

```bash
pip install wandb
```

首次使用需登录（需有 [wandb.ai](https://wandb.ai) 账号）：

```bash
wandb login
```

> 💡 **无外网环境**：跳过登录，直接使用 `mode="offline"`（见第3节）。

## 2. 基础用法：`wandb.init()`

在训练脚本开头初始化一个 run：

```python
import wandb

wandb.init(
    project="cs336-a5-sft-v2",          # 项目名（必选）
    entity="your-team-or-username",     # 团队/用户名（可选）
    name="wanda_sft",        # 可读 run 名称
    config={
        "model": "Qwen2.5-Math-1.5B",
        "dataset_tag": "raw", # raw, sf, grpo
        "batch_size": 64,
        "max_examples": "1000",
        "seed": 2026,
        "learning_rate": 2e-5,
    }
)
```

`config`参数可以自己定义，建议将所有超参数、数据路径、模型版本等放入 `config`，便于后续筛选与比较。


## 3. 离线模式（Offline Mode）

当服务器无法访问外网时，使用离线模式保存日志：

```python
wandb.init(mode="offline", ...)
```

所有日志将保存在本地 `wandb/` 目录下，格式为 `offline-run-<timestamp>-<id>`。

### 后续同步到云端

将包含 `wandb/` 目录的文件夹拷贝到有网络的机器，执行：

```bash
wandb sync wandb/
```

> ⚠️ 注意：确保该机器已 `wandb login`，且 run ID 未被删除。

你也可以只同步特定 run：

```bash
wandb sync wandb/offline-run-20260116_113519-vc1rtokn
```

---

## 4. 记录指标：`wandb.log()`

在训练/评估循环中记录标量、图像、文本等：

```python
for step, batch in enumerate(dataloader):
    loss = model(batch)
    wandb.log({
        "train/loss": loss.item(),
        "train/lr": scheduler.get_last_lr()[0],
        "step": step
    })
```

支持：
- 标量（scalar）
- 图像（`wandb.Image`）
- 文本（`wandb.Table`）
- 直方图（`wandb.Histogram`）
- 音频、3D 对象等（较少用于 LLM）

> 📌 **技巧**：使用 `/` 分隔命名空间（如 `eval/human_eval_pass@1`），便于 UI 中分组展示。


## 5. 保存模型与工件（Artifacts）

W&B 支持将模型 checkpoint 作为 **Artifact** 上传，实现版本控制：

```python
artifact = wandb.Artifact(name="llama3-70b-wanda-c4", type="model")
artifact.add_file("checkpoints/model.safetensors")
wandb.log_artifact(artifact)
```

后续可在其他实验中引用该模型：

```python
artifact = run.use_artifact("llama3-70b-wanda-c4:latest")
artifact_dir = artifact.download()
```

> 🔒 **注意**：大模型文件较大，不建议上传模型参数文件。


## 6. 常见问题

#### Q1: 初始化超时？
```python
wandb.errors.CommError: Run initialization has timed out...
```
**解决**：增加超时时间或切离线模式：
```python
wandb.init(settings=wandb.Settings(init_timeout=120), mode="offline")
```

#### Q2: 能否禁用 W&B（如调试时）？
```python
wandb.init(mode="disabled")  # 完全静默，不产生任何副作用
```

---

## 7. 参考材料

- 官方文档：https://docs.wandb.ai
- 示例仓库：https://github.com/wandb/examples

> ✨ **小贴士**：每次实验前花 2 分钟写好 `config` 和 `notes`，未来回溯时会感谢自己！
