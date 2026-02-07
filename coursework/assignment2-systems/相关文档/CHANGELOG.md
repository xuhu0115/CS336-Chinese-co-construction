# 更新日志（Changelog）

我们对作业代码或 PDF 所做的所有更改都会记录在此文件中。

## [1.0.5] - 2025-07-03
### 修复（Fixed）
- 代码：修复 adapters 中与 FlashAttention2 相关的拼写错误

### 新增（Added）
- 代码：新增 `.python-version` 约束，指定 Python 3.12

## [1.0.4] - 2025-04-27

### 修复（Fixed）
- 讲义：补充排行榜提交的相关细节
- 讲义：在 DDP 场景下，用 Nsight 替换 PyTorch profiler

### 新增（Added）
- 讲义：明确需要进行基准测试的 attention 实现

## [1.0.3] - 2025-04-24
### 修复（Fixed）
- 讲义：`memory_profiling` (b) 应当对上下文长度进行扫描，而不是模型大小
- 讲义：FA2 Triton 起始代码中的关键字参数由 `tile_shape` 改为 `block_shape`
- 讲义：`flash_backward` 题目描述中的一个小拼写错误
- 讲义：使用 `uv run` 启动 all-gather 示例

## [1.0.2] - 2025-04-22
### 新增（Added）
- 讲义：澄清 flash autograd 函数的接口
- 讲义：澄清 attention 基准测试的提交方式

### 修复（Fixed）
- 讲义：修复 flash 算法中的一些小符号问题
- 讲义：修复 flash backward 中关于节省开销的数学错误及解释
- 讲义：调整 attention 基准测试中使用的参数，使其更加合理

### 移除（Removed）
- 讲义：移除部分与现代 PyTorch 不兼容的内存基准测试

## [1.0.1] - 2025-04-17
### 新增（Added）
- 代码：为 flash forward 实现中的 logsumexp 添加测试
- 讲义：澄清 flash autograd 函数的接口
- 代码：测试 forward 和 backward 中的 `causal=True`

## [1.0.0] - 2025-04-16

### 新增（Added）
- 讲义 / 代码：加入 FlashAttention2
- 讲义：加入基于 Nsight Systems 的规范化性能分析
- 讲义：加入通信量统计（communication accounting）
- 代码：为额外内容添加测试

### 变更（Changed）
- 讲义：大幅改进 Triton 的示例演示
- 讲义：移除通信相关的重复性工作（busywork）

### 修复（Fixed）
- 讲义：澄清 `ddp_bucketed_benchmarking` 不需要完整的运行参数网格

## [0.0.4] - 2024-04-23

### 新增（Added）

### 变更（Changed）
- 代码：移除 DDP 测试中的 try-finally 代码块

### 修复（Fixed）
- 讲义：移除对一个并不存在于作业中的问题的过时描述
- 讲义：修复示例中的 Slurm 环境变量
- 讲义：澄清 `ddp_bucketed_benchmarking` (b) 中的假设条件

## [0.0.3] - 2024-04-21

### 新增（Added）

### 变更（Changed）
- 代码：从 requirements.txt 中移除 `humanfriendly`，新增 `matplotlib`
- 讲义：修改问题 `distributed_communication_multi_node`，明确多节点测量应为 2x1、2x2 和 2x3
- 讲义：澄清即使在 `async_op=False` 的情况下，计时 collective 通信操作也必须调用 `torch.cuda.synchronize()`

### 修复（Fixed）
- 讲义：修复 memory_profiling (a) 题目中被截断的文本
- 讲义：修复第 3.2 节中 slurm 配置与描述文本不一致的问题
- 代码：修复 `ToyModelWithTiedWeights`，使其真正共享权重
- 讲义：修复 bucketed DDP 测试命令中的拼写错误，应为 `pytest tests/test_ddp.py`
- 讲义：修复 `ddp_overlap_individual_parameters_benchmarking` (a) 的交付要求，不再要求通信时间，只需端到端 step 时间
- 讲义：澄清 `optimizer_state_sharding_accounting` (a) 中的分析说明

## [0.0.1] - 2024-04-17

### 新增（Added）
- 讲义：在 benchmarking_script 问题中加入一个关于结果波动性的简短问题

### 变更（Changed）

### 修复（Fixed）
- 讲义：修复 `triton_rmsnorm_forward` 题目中的拼写错误，adapters 应返回类本身，而不是 `.apply` 属性
- 代码：在 `./cs336-systems/'[test]'` 中新增 `-e` 参数
- 讲义：澄清关于 timeit 模块的推荐用法
- 讲义：澄清关于 CUDA 总时间最高的 kernel 的问题

## [0.0.0] - 2024-04-16

初始版本发布。
