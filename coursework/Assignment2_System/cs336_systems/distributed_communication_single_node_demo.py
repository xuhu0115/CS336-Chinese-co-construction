import os
import time
import torch
import torch.distributed as dist
import argparse
import torch
from torch.multiprocessing import spawn


# ============================================================
# 1. 分布式环境初始化与清理
# ============================================================

def init_distributed_environment(
    master_addr: str,
    master_port: int,
    global_rank: int,
    world_size: int,
    backend: str
):
    """
    初始化 PyTorch 分布式通信环境（进程组）

    参数说明：
    - master_addr : rank 0 所在节点的 IP 地址
    - master_port : 用于进程间通信的端口
    - global_rank : 当前进程在所有进程中的唯一编号
    - world_size  : 总进程数
    - backend     : 通信后端（gloo / nccl / mpi）
    """

    # --------------------------------------------------------
    # 1.1 设置 rendezvous（所有进程汇合的地址）
    # --------------------------------------------------------
    # 所有进程必须通过 MASTER_ADDR:MASTER_PORT 建立初始连接
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # --------------------------------------------------------
    # 1.2 初始化进程组（分布式的“总开关”）
    # --------------------------------------------------------
    # 这是使用 torch.distributed 的前置条件
    dist.init_process_group(
        backend=backend,
        rank=global_rank,
        world_size=world_size
    )


def destroy_distributed_environment():
    """
    清理分布式环境并释放资源
    """

    # 销毁进程组，防止资源泄漏或死锁
    dist.destroy_process_group()

    # GPU 场景下，清空 PyTorch 的 CUDA cache（非强制，但推荐）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 2. All-Reduce 通信性能 Benchmark
# ============================================================

def run_all_reduce_benchmark(
    global_rank: int,
    world_size: int,
    tensor_size_mb: int,
    backend: str,
    device: str,
    master_addr: str,
    master_port: int
):
    """
    对 dist.all_reduce 进行性能测试

    测试指标：
    - 单次 all-reduce 平均耗时
    - 理论通信带宽（GB/s）
    """

    # --------------------------------------------------------
    # 2.1 初始化分布式环境
    # --------------------------------------------------------
    init_distributed_environment(
        master_addr=master_addr,
        master_port=master_port,
        global_rank=global_rank,
        world_size=world_size,
        backend=backend
    )

    # --------------------------------------------------------
    # 2.2 绑定 GPU（仅在 CUDA 场景下）
    # --------------------------------------------------------
    if device == "cuda":
        # 通常约定：rank i → GPU i
        torch.cuda.set_device(global_rank)
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 2.3 构造通信测试用的 Tensor
    # --------------------------------------------------------
    # 将 MB 转换为字节
    tensor_num_bytes = tensor_size_mb * 1024 * 1024

    # float32 占 4 字节
    bytes_per_element = 4
    num_elements = tensor_num_bytes // bytes_per_element

    # 随机生成测试数据（数值本身不重要）
    communication_tensor = torch.randn(
        num_elements,
        device=device,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # 2.4 Warm-up（非常重要）
    # --------------------------------------------------------
    # 原因：
    # - NCCL 通信器初始化
    # - CUDA kernel lazy initialization
    # - GPU 频率爬升
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        dist.all_reduce(
            communication_tensor,
            op=dist.ReduceOp.SUM
        )

        # CUDA 是异步执行，必须显式同步
        if device == "cuda":
            torch.cuda.synchronize()

    # 确保所有进程在同一时刻开始正式测试
    dist.barrier()

    # --------------------------------------------------------
    # 2.5 正式 benchmark（计时）
    # --------------------------------------------------------
    benchmark_iterations = 20

    start_time = time.time()

    for _ in range(benchmark_iterations):
        dist.all_reduce(
            communication_tensor,
            op=dist.ReduceOp.SUM
        )

    # GPU 场景下等待所有 kernel 完成
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # --------------------------------------------------------
    # 2.6 性能指标计算
    # --------------------------------------------------------
    total_elapsed_time = end_time - start_time
    avg_latency_seconds = total_elapsed_time / benchmark_iterations

    # 带宽 = 数据量 / 时间（单位：GB/s）
    bandwidth_gbps = (tensor_num_bytes / avg_latency_seconds) / 1e9

    # --------------------------------------------------------
    # 2.7 打印结果（仅 rank 0）
    # --------------------------------------------------------
    if global_rank == 0:
        print(
            f"[All-Reduce Benchmark]\n"
            f"  Backend        : {backend}\n"
            f"  Device         : {device}\n"
            f"  World Size     : {world_size}\n"
            f"  Tensor Size    : {tensor_size_mb} MB\n"
            f"  Avg Latency    : {avg_latency_seconds * 1000:.4f} ms\n"
            f"  Bandwidth      : {bandwidth_gbps:.4f} GB/s\n"
        )

    # --------------------------------------------------------
    # 2.8 汇总所有 rank 的结果（教学用）
    # --------------------------------------------------------
    local_result = {
        "rank": global_rank,
        "world_size": world_size,
        "backend": backend,
        "device": device,
        "tensor_size_mb": tensor_size_mb,
        "avg_latency_ms": avg_latency_seconds * 1000,
        "bandwidth_gbps": bandwidth_gbps,
    }

    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    # --------------------------------------------------------
    # 2.9 清理环境
    # --------------------------------------------------------
    destroy_distributed_environment()

    # 只让主进程返回结果
    if global_rank == 0:
        return gathered_results


def main():

    for size in [1,10,100,1000]:
        for Backend in ["gloo", "nccl"]:
            for world_size in [1]:
                parser = argparse.ArgumentParser("All-Reduce Benchmark")

                parser.add_argument("--world_size", type=int, default=world_size,
                                    help="进程总数（通常等于 GPU 数）")
                parser.add_argument("--tensor_size_mb", type=int, default=size,
                                    help="通信 tensor 大小（MB）")
                parser.add_argument("--backend", type=str, default=Backend,
                                    choices=["gloo", "nccl"],
                                    help="分布式通信后端")
                parser.add_argument("--device", type=str, default="cuda",
                                    choices=["cpu", "cuda"],
                                    help="运行设备")
                parser.add_argument("--master_addr", type=str, default="127.0.0.1")
                parser.add_argument("--master_port", type=int, default=29500)

                args = parser.parse_args()

                # GPU 数量检查
                if args.device == "cuda":
                    assert torch.cuda.is_available(), "CUDA 不可用"
                    assert args.world_size <= torch.cuda.device_count(), (
                        "world_size 不能超过 GPU 数量"
                    )

                # 使用 spawn 启动多进程
                spawn(
                    fn=run_all_reduce_benchmark,
                    args=(
                        args.world_size,
                        args.tensor_size_mb,
                        args.backend,
                        args.device,
                        args.master_addr,
                        args.master_port
                    ),
                    nprocs=args.world_size,
                    join=True
                )


if __name__ == "__main__":
    main()

