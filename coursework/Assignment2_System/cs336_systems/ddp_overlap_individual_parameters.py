import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(SimpleNet, self).__init__()

        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        # Create the layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = torch.flatten(x, 1)

        # Forward through all layers except the last one with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Last layer without activation (for use with cross-entropy loss)
        x = self.layers[-1](x)

        return F.log_softmax(x, dim=1)

class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super(DDPOverlapBucketed, self).__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.handles = []
        self.buckets = []
        self.world_size = dist.get_world_size()
        # 初始参数广播：确保所有进程都有相同的初始参数
        self._broadcast_parameters()
        #创建类的时候就立即执行这两个函数
        self._create_bucket()
        self._reigster_hook()


    def _broadcast_parameters(self):
        """
        将rank 0的参数广播到所有其他进程，确保所有进程都有相同的初始参数
        """
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _create_bucket(self):
        current_bucket_size = 0
        current_bucket = []
        for p in reversed(list(self.module.parameters())):
            if p.requires_grad:
                p_size = p.numel() * p.element_size()
                if p_size + current_bucket_size > self.bucket_size_bytes and current_bucket:#当前的桶已经满了，保存当前的桶，并创建新的桶,中间变量清0
                    self.buckets.append(current_bucket)
                    current_bucket_size = 0
                    current_bucket = []
                current_bucket.append(p)
                current_bucket_size += p_size
        #最后如果还有剩余的桶没满，也保存到桶s中
        if current_bucket:  # 只有当桶不为空时才添加
            self.buckets.append(current_bucket)

        for i, bucket_params in enumerate(self.buckets):
            if not bucket_params:  # 检查桶是否为空
                continue
            #创建一个缓冲区，记录每一个桶的梯度的元素的数量
            buffer_size = sum(p.numel() for p in bucket_params)
            buffer = torch.zeros(buffer_size, device=bucket_params[0].device, dtype=bucket_params[0].dtype)
            self.buckets[i] = {
                "params": bucket_params,
                "buffer": buffer,
                "ready_params": set(),
                "triggered": False
            }



    def _reigster_hook(self):
        for bucket_idx, bucket_info in enumerate(self.buckets):
            for param in bucket_info["params"]:
                # 创建一个闭包来捕获正确的bucket_idx值
                def make_hook(idx):
                    return lambda grad, param=param: self._create_hook(grad, param, idx)
                param.register_hook(make_hook(bucket_idx))

    def _create_hook(self, grad, param, bucket_idx):
        bucket_info = self.buckets[bucket_idx]
        bucket_info["ready_params"].add(param)
        if len(bucket_info["ready_params"]) == len(bucket_info["params"]) and not bucket_info["triggered"]:
            # 标记该桶已经触发通信，防止重复触发
            bucket_info["triggered"] = True

            # 延迟执行，确保所有梯度都已计算完成
            def delayed_sync():
                #  核心逻辑：将桶内所有梯度拷贝到扁平的缓冲区 
                offset = 0
                for p in bucket_info["params"]:
                    numel = p.numel()
                    if p.grad is not None:
                        bucket_info["buffer"][offset:offset+numel].copy_(p.grad.view(-1))
                    else:
                        # 如果没有梯度，填充零
                        bucket_info["buffer"][offset:offset+numel].zero_()
                    offset += numel

                #  核心逻辑：启动异步的AllReduce通信 
                # 这允许通信和后续的梯度计算重叠
                handle = dist.all_reduce(bucket_info["buffer"], async_op=True)
                self.handles.append((handle, bucket_idx))

            # 使用torch.autograd.Variable的hook机制来延迟执行
            # 这确保在所有梯度计算完成后再执行同步
            import torch.autograd as autograd
            autograd.Variable._execution_engine.queue_callback(delayed_sync)
    def forward(self, x):
        #每次forward之前，清空桶的触发状态和ready_params
        if self.world_size > 1:
            for bucket in self.buckets:
                bucket["triggered"] = False
                bucket["ready_params"].clear()
        self.handles.clear()
        return self.module(x)

    def finish_gradient_synchronization(self):
        """
        等待所有排队的异步通信完成。
        这个函数应该在optimizer.step()之前被调用。
        """
        # 等待所有通信句柄完成
        for handle, bucket_idx in self.handles:
            handle.wait()

            #  核心逻辑：将通信完成后的梯度从缓冲区写回每个参数 
            bucket_info = self.buckets[bucket_idx]
            buffer = bucket_info["buffer"]
            # AllReduce求和后需要除以world_size得到平均值
            buffer.div_(self.world_size)

            offset = 0
            for p in bucket_info["params"]:
                numel = p.numel()
                # 确保param.grad存在，然后用缓冲区的数据覆盖它
                if p.grad is not None:
                    p.grad.view(-1).copy_(buffer[offset:offset+numel])#将缓冲区的数据写回每个参数
                offset += numel

        # 清除handles，为下一次迭代做准备
        self.handles.clear()


import os, torch, torch.distributed as dist, torch.nn as nn

def run():
    dist.init_process_group(backend="nccl")   # torchrun 已把环境变量配好
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = nn.Linear(10, 5).cuda()
    ddp_model = DDPOverlapBucketed(model, bucket_size_mb=1)   # 1 MB 小桶方便观察
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    for step in range(3):
        x = torch.randn(4, 10).cuda()
        loss = ddp_model(x).sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()   # 必须手动调
        opt.step()
        opt.zero_grad()
        print(f"rank {dist.get_rank()} step {step} loss={loss.item()}")

if __name__ == "__main__":
    run()
