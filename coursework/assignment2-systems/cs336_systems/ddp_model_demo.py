
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.multiprocessing.spawn import spawn


# ============================
# 一、定义一个简单的全连接神经网络
# ============================
class SimpleNet(nn.Module):
    """
    一个最简单的三层全连接网络，用于 MNIST 分类
    输入：28×28 展平后的 784 维向量
    输出：10 类（0~9）
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        # 第一层全连接：输入层 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 第二层全连接：隐藏层 -> 隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层：隐藏层 -> 类别数
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播逻辑
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 使用 log_softmax，方便后续配合 NLLLoss
        return F.log_softmax(x, dim=1)


# ============================
# 二、分布式环境初始化与清理
# ============================
def init_distributed_env(rank: int, world_size: int, backend: str):
    """
    初始化分布式进程组（所有进程都必须调用）
    """
    # 主进程地址和端口
    # 在单机多卡场景中，localhost 即可
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def destroy_distributed_env():
    """
    销毁进程组并清理资源
    """
    dist.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================
# 三、DDP 训练主逻辑（教学重点）
# ============================
def ddp_training_worker(rank: int, world_size: int, backend: str):
    """
    每一个进程都会执行这个函数
    rank      : 当前进程编号
    world_size: 总进程数
    """

    # ------------------------------------------------
    # 1. 固定随机种子（确保所有进程行为一致）
    # ------------------------------------------------
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 确保 cudnn 的确定性行为（教学更容易对齐结果）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------
    # 2. 初始化分布式环境
    # ------------------------------------------------
    init_distributed_env(rank, world_size, backend)

    # ------------------------------------------------
    # 3. 为当前进程分配设备
    # ------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"[进程 {rank}] 使用 GPU：{device}")
    else:
        device = torch.device("cpu")
        print(f"[进程 {rank}] CUDA 不可用，使用 CPU")

    # ------------------------------------------------
    # 4. 数据预处理与加载
    # ------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # 将 28×28 展平为 784
        transforms.Lambda(lambda x: x.view(-1))
    ])

    full_dataset = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    # ------------------------------------------------
    # 5. 手动划分数据（教学版 DDP）
    #    每个进程只处理数据集的一部分
    # ------------------------------------------------
    total_samples = len(full_dataset)
    samples_per_rank = total_samples // world_size

    start_index = rank * samples_per_rank
    end_index = start_index + samples_per_rank

    subset_dataset = torch.utils.data.Subset(
        full_dataset,
        range(start_index, end_index)
    )

    # DataLoader 使用相同随机种子，确保 shuffle 行为可复现
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=64,
        shuffle=True,
        generator=data_generator
    )

    print(
        f"[进程 {rank}] 负责样本区间："
        f"{start_index} ~ {end_index - 1}，"
        f"共 {samples_per_rank} 条数据"
    )

    # ------------------------------------------------
    # 6. 创建模型与优化器
    # ------------------------------------------------
    model = SimpleNet(
        input_dim=784,
        hidden_dim=50,
        num_classes=10
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------
    # 7. 步骤一：广播模型参数（关键教学点）
    #    确保所有进程从“完全相同”的初始模型开始
    # ------------------------------------------------
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model.train()

    # ------------------------------------------------
    # 8. 正式开始训练
    # ------------------------------------------------
    for epoch in range(1, 3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # ----------------------------
            # 步骤二：前向 + 反向传播
            # 每个进程只计算自己那一部分数据的梯度
            # ----------------------------
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)

            step_start_time = time.time()
            loss.backward()

            # ----------------------------
            # 步骤三：梯度 All-Reduce
            # 将所有进程的梯度相加，再取平均
            # ----------------------------
            comm_start_time = time.time()

            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(
                        param.grad.data,
                        op=dist.ReduceOp.SUM
                    )
                    param.grad.data /= world_size

            comm_time = time.time() - comm_start_time

            # ----------------------------
            # 步骤四：参数更新
            # 所有进程使用“完全相同”的平均梯度
            # ----------------------------
            optimizer.step()

            step_time = time.time() - step_start_time

            if batch_idx % 50 == 0:
                print(
                    f"[进程 {rank}] "
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx} | "
                    f"Loss = {loss.item():.6f}"
                )

            if rank == 0 and batch_idx % 50 == 0:
                print(
                    f"[主进程] 单步耗时：{step_time * 1000:.2f} ms，"
                    f"通信耗时：{comm_time * 1000:.2f} ms"
                )

    # ------------------------------------------------
    # 9. 仅在 rank 0 保存模型
    # ------------------------------------------------
    if rank == 0:
        torch.save(model.state_dict(), "mnist_simple_ddp.pt")
        print("[主进程] 模型已保存：mnist_simple_ddp.pt")

    destroy_distributed_env()


# ============================
# 四、程序入口
# ============================
def main():
    # world_size = 使用的进程（GPU）数量
    # 如果只有一张 GPU，就设为 1（等价于普通训练）
    world_size = 1

    # NCCL：GPU 通信首选后端
    backend = "nccl"

    # 检查数据是否能被平均分配
    dataset_size = len(
        datasets.MNIST("../data", train=True, download=True)
    )

    if dataset_size % world_size != 0:
        print(
            f"警告：数据量 {dataset_size} 不能被 world_size={world_size} 整除，"
            f"将丢弃部分样本以保证均分"
        )

    print(f"启动分布式训练，进程数：{world_size}")
    print(f"每个进程处理样本数：{dataset_size // world_size}")

    # 启动多进程
    spawn(
        ddp_training_worker,
        args=(world_size, backend),
        nprocs=world_size,
        join=True
    )

    print("分布式训练结束")


if __name__ == "__main__":
    main()
