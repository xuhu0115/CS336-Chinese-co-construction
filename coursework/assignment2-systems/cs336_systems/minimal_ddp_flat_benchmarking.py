import torch
import torch.distributed as dist
import os
from torchvision import datasets, transforms
import torch.optim as optim
import time
from torch.multiprocessing.spawn import spawn
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        # 定义第一个隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义第二个隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义输出层
        self.fc3 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  
# 示例：创建一个SimpleNet实例

def setup(rank, world_size, backend):
    """ 初始化分布式环境 """
    os.environ['MASTER_ADDR'] = 'localhost' # 设置主节点IP地址以及端口，其他节点需要通过这个地址连接到主节点
    os.environ['MASTER_PORT'] = '29500'
    # 根据后端初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size) # 初始化进程组，rank是当前进程的rank，world_size是总进程数，backend: 这是指定通信后端的参数。常见的后端有：gloo(CPU), nccl(GPU), mpi等。

def cleanup():
    """ 清理分布式环境 """
    dist.destroy_process_group()
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ddp_train(rank, world_size, backend):
    """ Naive DDP训练函数，实现图片中描述的4个步骤 """
    # 设置随机种子确保可重现性（所有进程使用相同种子）
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 初始化分布式环境
    setup(rank, world_size, backend)

    # 为每个进程分配不同的GPU设备
    # 使用GPU 0和1，它们当前是空闲的
    gpu_id = rank  # rank 0 -> GPU 0, rank 1 -> GPU 1
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Rank {rank} using GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU for rank {rank}")

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    # 为了验证DDP正确性，我们让每个进程处理不同的数据子集
    # 这样总的有效batch size = 64 * world_size
    total_samples = len(dataset)
    samples_per_process = total_samples // world_size
    start_idx = rank * samples_per_process
    end_idx = start_idx + samples_per_process

    # 创建当前进程的数据子集
    subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))

    # 设置确定性的数据加载器
    # 重要：为了与单进程训练比较，我们需要确保数据顺序一致
    generator = torch.Generator()
    generator.manual_seed(seed)  # 所有进程使用相同的种子确保一致性
    train_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, generator=generator)

    print(f"Rank {rank} processing samples {start_idx} to {end_idx-1} ({samples_per_process} samples)")
    print(f"Rank {rank} effective batch size: 64, total distributed batch size: {64 * world_size}")

    # 创建模型和优化器（每个设备都创建相同的模型）
    model = SimpleNet(input_size=784, hidden_size=50, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # ===== 新增：记录梯度展平所需的元信息 =====
    grad_shapes = []
    grad_numels = []

    for p in model.parameters():
        grad_shapes.append(p.shape)
        grad_numels.append(p.numel())

    total_grad_numel = sum(grad_numels)

    # 步骤1: Broadcast模型参数从rank 0到所有其他ranks
    # 确保所有设备从相同的初始模型和优化器状态开始
    for param in model.parameters():
        # 确保参数在正确的设备上进行broadcast
        dist.broadcast(param.data, src=0)

    # 同步优化器状态（如果有的话）
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                for key, value in optimizer.state[param].items():
                    if torch.is_tensor(value):
                        dist.broadcast(value, src=0)

    model.train()

    for epoch in range(1, 3):  # Train for 2 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 步骤2: 每个设备使用本地模型参数进行前向传播和反向传播
            # 计算n/d个样本的梯度
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            step_start = time.time()
            loss.backward()

            comm_start = time.time()
            # 步骤3: All-reduce梯度
            # 将所有设备的梯度求平均，使每个设备都持有所有n个样本的平均梯度

            # ===== 新的步骤3：Flattened gradient all-reduce =====
            flat_grad = torch.zeros(total_grad_numel, device=device)

            offset = 0
            for p in model.parameters():
                if p.grad is not None:
                    numel = p.grad.numel()
                    flat_grad[offset:offset + numel] = p.grad.view(-1)
                    offset += numel

            # 单次通信
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad /= world_size

            # 写回每个参数的梯度
            offset = 0
            for p, shape, numel in zip(model.parameters(), grad_shapes, grad_numels):
                if p.grad is not None:
                    p.grad.copy_(
                        flat_grad[offset:offset + numel].view(shape)
                    )
                    offset += numel

            comm_time = time.time() - comm_start
            # 步骤4: 优化器步骤 - 每个设备使用相同的平均梯度更新参数
            # 由于所有设备从相同初始状态开始并使用相同梯度，参数会保持同步
            optimizer.step()
            step_time = time.time() - step_start
            if batch_idx % 50 == 0:
                print(f'Rank {rank}, Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Step time: {step_time*1000:.2f} ms | Comm time: {comm_time*1000:.2f} ms")

    # 只在rank 0保存模型
    if rank == 0:
        torch.save(model.state_dict(), "mnist_simple_ddp.pt")
        print("Model saved!")

    cleanup()

def main():
    world_size = 1 # gpu只有一个就改成1，但是这样和平时普通的训练就没有区别了
    backend = 'nccl'  # 使用gloo后端，更适合CPU训练
    # 只检查数据集大小是否能被world_size整除
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    data_size = len(dataset)

    if data_size % world_size != 0:
        print(f"Warning: Data size {data_size} is not divisible by world size {world_size}")
        print(f"Some samples will be ignored to ensure equal distribution")

    print(f"Starting distributed training with {world_size} processes")
    print(f"Each process will handle {data_size // world_size} samples")
    print(f"Total samples: {data_size}")

    # 启动分布式训练
    spawn(ddp_train, args=(world_size, backend), nprocs=world_size, join=True)

    print("Distributed training completed!")

if __name__ == '__main__':
    main()
