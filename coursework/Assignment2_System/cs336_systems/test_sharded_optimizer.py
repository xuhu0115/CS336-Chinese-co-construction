import os
import torch
import torch.nn as nn
import torch.distributed as dist

from sharded_optimizer import ShardedOptimizer   # 假设你把之前代码存成这个文件


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    setup_distributed()

    rank = dist.get_rank()
    device = torch.device("cuda")

    # 一个简单模型
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device)

    # 确保模型初始化一致
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = ShardedOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    for step in range(5):
        x = torch.randn(8, 1024, device=device)
        y = torch.randint(0, 10, (8,), device=device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
