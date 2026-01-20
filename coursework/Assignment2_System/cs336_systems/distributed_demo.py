# =========================
# 标准库 & PyTorch 相关导入
# =========================

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# =========================
# 分布式环境初始化函数
# =========================
def setup(rank, world_size):
    """
    初始化分布式通信环境（Process Group）

    参数：
    - rank: 当前进程的编号（0 ~ world_size-1）
    - world_size: 总进程数
    """

    # 指定“主进程”的地址
    # 所有进程都会通过这个地址进行 rendezvous（集合）
    # 单机多进程时使用 localhost
    os.environ["MASTER_ADDR"] = "localhost"

    # 指定主进程监听的端口号
    # 所有进程必须保持一致
    os.environ["MASTER_PORT"] = "29500"

    # 初始化进程组（非常关键的一步）
    # backend="gloo"：使用 gloo 通信后端（CPU 通用，支持多平台）
    # rank：当前进程在所有进程中的唯一编号
    # world_size：参与通信的总进程数
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )


# =========================
# 每个进程实际执行的函数
# =========================
def distributed_demo(rank, world_size):
    """
    每个进程都会执行这个函数一次
    """

    # 初始化分布式通信环境
    setup(rank, world_size)

    # 每个进程各自生成一个本地张量
    # torch.randint 是“本地操作”，不同 rank 得到的值不同
    # 这里生成一个 shape 为 (3,) 的一维张量
    data = torch.randint(0, 10, (3,))

    # 打印 all-reduce 之前的数据
    # 注意：由于是多进程并行执行，打印顺序不确定
    print(f"rank {rank} data (before all-reduce): {data}")

    # 核心操作：all-reduce
    # - 对所有 rank 上的 data 做规约（默认是 SUM）
    # - 结果会“原地”写回到每个 rank 的 data 中
    # - async_op=False 表示同步执行（阻塞，直到完成）
    dist.all_reduce(data, async_op=False)

    # 打印 all-reduce 之后的数据
    # 此时每个 rank 的 data 都是完全相同的
    print(f"rank {rank} data (after all-reduce): {data}")


# =========================
# 程序主入口
# =========================
if __name__ == "__main__":

    # world_size 表示总进程数
    # 在 DDP 中通常等于 GPU 数量
    world_size = 4

    # 使用 torch.multiprocessing.spawn 启动多进程
    mp.spawn(
        fn=distributed_demo,   # 每个进程要执行的函数
        args=(world_size,),    # 传给函数的额外参数（rank 会自动作为第一个参数）
        nprocs=world_size,     # 启动的进程数量
        join=True              # 主进程是否等待所有子进程结束
    )
