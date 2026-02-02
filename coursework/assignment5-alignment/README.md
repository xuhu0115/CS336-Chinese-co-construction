# CS336 Spring 2025 Assignment 5: Alignment

有关作业的完整说明，请参阅作业讲义：[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

我们提供一份关于安全对齐、指令微调和 RLHF 的补充（完全可选）作业 [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

如果发现讲义或代码有问题，欢迎提交 GitHub issue 或 pull request 进行修复。

## 环境配置

如同之前的作业，我们使用 `uv` 管理依赖。

1. 先安装除 flash-attn 之外的所有包，然后再安装全部包（因为 flash-attn 比较特殊）：
```
uv sync --no-install-package flash-attn
uv sync
```

2. 运行单元测试：

``` sh
uv run pytest
```

开始时，所有测试都会因为 NotImplementedError 而失败。要让你的实现接入测试，请完成[./tests/adapters.py](./tests/adapters.py)中的函数。 

