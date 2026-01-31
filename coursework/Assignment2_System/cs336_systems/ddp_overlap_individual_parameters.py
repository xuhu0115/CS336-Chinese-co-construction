import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

# ============================
# 二、手动 DDP（按桶通信梯度）
# ============================
class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super(DDPOverlapBucketed, self).__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.handles = []   # 保存异步 all_reduce 句柄
        self.buckets = []   # 梯度桶
        self.world_size = dist.get_world_size()

        # --------------------
        # 初始化：广播参数 + 创建桶 + 注册钩子
        # --------------------
        self._broadcast_parameters()  # 初始参数广播
        self._create_bucket()         # 按大小划分梯度桶
        self._register_hook()         # 注册梯度钩子实现异步通信

    # --------------------
    # 广播初始参数
    # --------------------
    def _broadcast_parameters(self):
        """
        将 rank 0 的参数广播到所有进程，确保初始状态一致
        """
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    # --------------------
    # 按 bucket_size 创建梯度桶
    # --------------------
    def _create_bucket(self):
        current_bucket_size = 0
        current_bucket = []

        # 倒序遍历参数，构建桶
        for p in reversed(list(self.module.parameters())):
            if p.requires_grad:
                p_size = p.numel() * p.element_size()

                # 当前桶满了则保存并创建新桶
                if p_size + current_bucket_size > self.bucket_size_bytes and current_bucket:
                    self.buckets.append(current_bucket)
                    current_bucket_size = 0
                    current_bucket = []

                current_bucket.append(p)
                current_bucket_size += p_size

        # 如果还有剩余参数，作为最后一个桶
        if current_bucket:
            self.buckets.append(current_bucket)

        # 为每个桶创建缓冲区
        for i, bucket_params in enumerate(self.buckets):
            if not bucket_params:
                continue
            buffer_size = sum(p.numel() for p in bucket_params)
            buffer = torch.zeros(buffer_size, device=bucket_params[0].device, dtype=bucket_params[0].dtype)
            self.buckets[i] = {
                "params": bucket_params,
                "buffer": buffer,
                "ready_params": set(),
                "triggered": False
            }

    # --------------------
    # 注册梯度钩子
    # --------------------
    def _register_hook(self):
        for bucket_idx, bucket_info in enumerate(self.buckets):
            for param in bucket_info["params"]:
                # 闭包捕获当前 bucket_idx
                def make_hook(idx):
                    return lambda grad, param=param: self._create_hook(grad, param, idx)
                param.register_hook(make_hook(bucket_idx))

    # --------------------
    # 梯度钩子逻辑：延迟执行 all_reduce
    # --------------------
    def _create_hook(self, grad, param, bucket_idx):
        bucket_info = self.buckets[bucket_idx]
        bucket_info["ready_params"].add(param)

        # 当桶内所有参数梯度都准备好，且未触发通信
        if len(bucket_info["ready_params"]) == len(bucket_info["params"]) and not bucket_info["triggered"]:
            bucket_info["triggered"] = True  # 标记为已触发

            def delayed_sync():
                # 将桶内所有梯度拷贝到扁平缓冲区
                offset = 0
                for p in bucket_info["params"]:
                    numel = p.numel()
                    if p.grad is not None:
                        bucket_info["buffer"][offset:offset+numel].copy_(p.grad.view(-1))
                    else:
                        bucket_info["buffer"][offset:offset+numel].zero_()
                    offset += numel

                # 启动异步 all_reduce，实现计算与通信重叠
                handle = dist.all_reduce(bucket_info["buffer"], async_op=True)
                self.handles.append((handle, bucket_idx))

            # 延迟执行，确保所有梯度计算完成
            import torch.autograd as autograd
            autograd.Variable._execution_engine.queue_callback(delayed_sync)

    # --------------------
    # forward 前清理状态
    # --------------------
    def forward(self, x):
        if self.world_size > 1:
            for bucket in self.buckets:
                bucket["triggered"] = False
                bucket["ready_params"].clear()
        self.handles.clear()
        return self.module(x)

    # --------------------
    # 等待所有异步通信完成
    # --------------------
    def finish_gradient_synchronization(self):
        """
        等待所有 all_reduce 完成，并将梯度写回各参数
        需在 optimizer.step() 之前调用
        """
        for handle, bucket_idx in self.handles:
            handle.wait()  # 等待通信完成

            bucket_info = self.buckets[bucket_idx]
            buffer = bucket_info["buffer"]

            # 求平均梯度
            buffer.div_(self.world_size)

            offset = 0
            for p in bucket_info["params"]:
                numel = p.numel()
                if p.grad is not None:
                    p.grad.view(-1).copy_(buffer[offset:offset+numel])
                offset += numel

        # 清空 handles，为下一次迭代准备
        self.handles.clear()


# ============================
# 三、简单示例运行
# ============================
def run():
    # 初始化分布式环境（torchrun 已设置环境变量）
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = nn.Linear(10, 5).cuda()
    ddp_model = DDPOverlapBucketed(model, bucket_size_mb=1)  # 1 MB 小桶方便观察
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    for step in range(3):
        x = torch.randn(4, 10).cuda()
        loss = ddp_model(x).sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()  # 必须手动调用
        opt.step()
        opt.zero_grad()
        print(f"[rank {dist.get_rank()}] step {step} loss={loss.item()}")


if __name__ == "__main__":
    run()
