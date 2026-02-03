from torch.profiler import ProfilerActivity
import torch
import os
from typing import Callable
import time
def profile(
    description: str,
    run: Callable,
    num_warmups: int = 1,
    with_stack: bool = False,
):
    """
    使用 PyTorch Profiler 对指定函数进行性能分析（CPU + CUDA）

    参数说明：
    - description:
        本次 profiling 的描述信息，用于区分不同实验
    - run:
        被 profiling 的函数（通常是一次完整的训练 step / forward-backward）
    - num_warmups:
        预热次数，用于消除首次运行带来的额外开销
    - with_stack:
        是否记录 CUDA 调用的 Python/C++ 堆栈信息（用于火焰图分析）
    """

    # =========================
    # 1 预热阶段（Warmup）
    # =========================

    # 预热的目的：
    # - 触发 CUDA kernel 的 JIT / lazy init
    # - 初始化 cuDNN / allocator
    # - 避免首次运行严重拉高 profiling 数据
    for _ in range(num_warmups):
        run()

    # =========================
    # 2 确认运行设备并同步
    # =========================

    if torch.cuda.is_available():
        print("正在使用 cuda")

        # CUDA 操作是异步的
        # 在 profiling 或计时前必须 synchronize，
        # 否则会统计到尚未完成的 kernel
        torch.cuda.synchronize()
    else:
        print("正在使用 cpu")

    # =========================
    # 3 使用 PyTorch Profiler 进行性能分析
    # =========================

    with torch.profiler.profile(
        # 指定需要分析的设备类型
        # 同时开启 CPU 和 CUDA
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],

        # 是否记录 Python / C++ 的调用栈
        # 用于后续生成火焰图或 stack trace
        with_stack=with_stack,

        # 实验性配置：
        # verbose=True 会输出更详细的 profiling 信息
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True
        ),
    ) as prof:

        # 在 profiler 作用域内运行被测函数
        run()

        # 再次同步，确保所有 CUDA kernel 完全执行完毕
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # =========================
    # 4 生成并格式化 profiling 结果表格
    # =========================

    table = prof.key_averages().table(
        # 按 CUDA 自身耗时排序（不包含子算子）
        sort_by="self_cuda_time_total",

        # 算子名称列的最大宽度
        max_name_column_width=80,

        # 最多显示的行数
        row_limit=10,

        # 是否只显示顶层算子
        # False 表示展开所有子算子
        top_level_events_only=False,
    )

    # =========================
    # 5 导出 CUDA 调用栈（用于可视化）
    # =========================

    if with_stack:
        # 创建输出目录
        os.makedirs("var", exist_ok=True)

        # 文本形式的 stack trace
        text_path = f"var/stacks_{description}.txt"

        # 火焰图（可用于 speedscope / flamegraph）
        svg_path = f"var/stacks_{description}.svg"

        # 按 self_cuda_time_total 导出 stack
        prof.export_stacks(text_path, "self_cuda_time_total")

    # 返回 profiling 表格（字符串）
    return table

def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # 热身：第一次运行可能较慢,因为要编译和缓存
    # 我们将多次要运行内核，因为重要的是稳态的运行时间。
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA 线程完成（非常重要！）
    print('现在真正计时!')
    times: list[float] = [] # @inspect times, @inspect description

    for trial in range(num_trials):  # 多次重复
        start_time = time.time()
        run()  # 实际执行计算
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待 CUDA 线程 完成同步
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times
    mean_time = mean(times) # 多次测量取平均

    print(f'单次耗时：{mean_time  }ms')
    return mean_time 