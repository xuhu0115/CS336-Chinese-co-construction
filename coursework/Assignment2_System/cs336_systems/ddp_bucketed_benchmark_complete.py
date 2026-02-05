import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.profiler
from torch.multiprocessing import spawn, Manager
import argparse
from typing import List, Dict
import json
import tempfile
import pickle

# ============================================================
# 1. 无分桶DDP实现（用于对比）
# ============================================================

class DDPNoBucket(nn.Module):
    """
    无分桶DDP：所有参数作为一个整体进行all_reduce
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        self._broadcast_parameters()
        self._register_hook()

    def _broadcast_parameters(self):
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _register_hook(self):
        self.params_with_grad = []
        for param in self.module.parameters():
            if param.requires_grad:
                self.params_with_grad.append(param)
                param.register_hook(lambda grad, p=param: self._hook_callback(p))

    def _hook_callback(self, param):
        if all(p.grad is not None for p in self.params_with_grad):
            flat_grads = []
            for p in self.params_with_grad:
                flat_grads.append(p.grad.view(-1))

            if flat_grads:
                buffer = torch.cat(flat_grads)
                handle = dist.all_reduce(buffer, async_op=True)
                self.handles.append((handle, buffer, self.params_with_grad))

    def forward(self, x):
        self.handles.clear()
        for p in self.params_with_grad:
            p.grad = None
        return self.module(x)

    def finish_gradient_synchronization(self):
        for handle, buffer, params in self.handles:
            handle.wait()
            buffer.div_(self.world_size)

            offset = 0
            for p in params:
                numel = p.numel()
                p.grad.view(-1).copy_(buffer[offset:offset+numel])
                offset += numel
        self.handles.clear()


# ============================================================
# 2. 分桶DDP实现（优化版）
# ============================================================

class DDPBucketed(nn.Module):
    """
    分桶DDP：按指定桶大小分组梯度，实现计算通信重叠
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.world_size = dist.get_world_size()
        self.handles = []
        self.buckets = []

        self._broadcast_parameters()
        self._create_buckets()
        self._register_hooks()

    def _broadcast_parameters(self):
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _create_buckets(self):
        """按桶大小创建梯度桶"""
        current_bucket = []
        current_bucket_size = 0

        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue

            p_size = p.numel() * p.element_size()

            if current_bucket and (current_bucket_size + p_size > self.bucket_size_bytes):
                self._finalize_bucket(current_bucket)
                current_bucket = []
                current_bucket_size = 0

            current_bucket.append(p)
            current_bucket_size += p_size

        if current_bucket:
            self._finalize_bucket(current_bucket)

    def _finalize_bucket(self, params: List[torch.nn.Parameter]):
        """为桶创建缓冲区"""
        if not params:
            return

        buffer_size = sum(p.numel() for p in params)
        buffer = torch.zeros(
            buffer_size, 
            device=params[0].device, 
            dtype=params[0].dtype
        )

        self.buckets.append({
            "params": params,
            "buffer": buffer,
            "ready_count": 0,
            "triggered": False,
            "total_params": len(params)
        })

    def _register_hooks(self):
        """为每个参数注册梯度钩子"""
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket["params"]:
                param.register_hook(
                    lambda grad, b_idx=bucket_idx: self._on_gradient_ready(b_idx)
                )

    def _on_gradient_ready(self, bucket_idx: int):
        """梯度就绪回调"""
        bucket = self.buckets[bucket_idx]
        bucket["ready_count"] += 1

        if (bucket["ready_count"] == bucket["total_params"] and 
            not bucket["triggered"]):

            bucket["triggered"] = True

            def launch_all_reduce():
                offset = 0
                for p in bucket["params"]:
                    numel = p.numel()
                    if p.grad is not None:
                        bucket["buffer"][offset:offset+numel].copy_(p.grad.view(-1))
                    else:
                        bucket["buffer"][offset:offset+numel].zero_()
                    offset += numel

                handle = dist.all_reduce(bucket["buffer"], async_op=True)
                self.handles.append((handle, bucket_idx))

            torch.autograd.Variable._execution_engine.queue_callback(launch_all_reduce)

    def forward(self, x):
        """前向传播，重置状态"""
        for bucket in self.buckets:
            bucket["triggered"] = False
            bucket["ready_count"] = 0

        self.handles.clear()
        return self.module(x)

    def finish_gradient_synchronization(self):
        """等待所有通信完成并写回梯度"""
        for handle, bucket_idx in self.handles:
            handle.wait()

            bucket = self.buckets[bucket_idx]
            bucket["buffer"].div_(self.world_size)

            offset = 0
            for p in bucket["params"]:
                numel = p.numel()
                if p.grad is not None:
                    p.grad.view(-1).copy_(bucket["buffer"][offset:offset+numel])
                offset += numel

        self.handles.clear()


# ============================================================
# 3. 0.5B模型定义
# ============================================================

class XLModel(nn.Module):
    """
    模拟XL大小的模型
    总参数量约 0.5B（5亿参数）
    """
    def __init__(self, hidden_size: int = 1024, num_layers: int = 24):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding层: 50000 * 1024 = 51.2M
        self.embedding = nn.Embedding(50000, hidden_size)

        # Transformer层: 24层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # 输出层: 1024 * 50000 = 51.2M
        self.output = nn.Linear(hidden_size, 50000)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# ============================================================
# 4. 基准测试主逻辑
# ============================================================

def run_benchmark(
    global_rank: int,
    world_size: int,
    bucket_size_mb: float,
    use_bucket: bool,
    num_iterations: int,
    master_addr: str,
    master_port: int,
    result_queue=None,
    result_file=None
):
    """
    运行DDP基准测试
    """

    # 设置设备
    torch.cuda.set_device(global_rank)
    device = torch.device(f"cuda:{global_rank}")

    # 初始化分布式环境 - 指定device_id消除警告
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # 修复：使用device_id指定设备，避免NCCL猜测
    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        device_id=device  # 关键修复：明确指定设备ID
    )

    if global_rank == 0:
        config_str = f"{'Bucketed' if use_bucket else 'No Bucket'}"
        if use_bucket:
            config_str += f" (bucket={bucket_size_mb}MB)"
        print(f"\n{'='*60}")
        print(f"Testing: {config_str}")
        print(f"World size: {world_size}")
        print(f"{'='*60}")

    model = XLModel().to(device)

    # 包装DDP
    if use_bucket:
        ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    else:
        ddp_model = DDPNoBucket(model)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # 预热
    if global_rank == 0:
        print("Warming up...")

    for _ in range(3):
        dummy_input = torch.randint(0, 50000, (2, 512), device=device)
        output = ddp_model(dummy_input)
        loss = output.mean()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

    # 修复：使用device参数明确指定barrier设备
    dist.barrier(device_ids=[global_rank])
    torch.cuda.synchronize()

    # 正式测试
    if global_rank == 0:
        print(f"Running benchmark ({num_iterations} iterations)...")

    iteration_times = []

    for iter_idx in range(num_iterations):
        input_ids = torch.randint(0, 50000, (2, 512), device=device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        output = ddp_model(input_ids)
        loss = output.mean()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_ms)

    # 收集所有rank的结果
    local_result = {
        "rank": global_rank,
        "bucket_size_mb": bucket_size_mb if use_bucket else float('inf'),
        "use_bucket": use_bucket,
        "iteration_times_ms": iteration_times,
        "avg_time_ms": sum(iteration_times) / len(iteration_times),
        "min_time_ms": min(iteration_times),
        "max_time_ms": max(iteration_times),
    }

    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_result)

    # PyTorch Profiler（仅rank 0，且仅分桶模式）
    if global_rank == 0 and use_bucket and bucket_size_mb <= 100:
        try:
            os.makedirs("./profiler_log", exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./profiler_log/bucket_{bucket_size_mb}MB"
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False
            ) as prof:

                for _ in range(4):
                    input_ids = torch.randint(0, 50000, (2, 512), device=device)
                    output = ddp_model(input_ids)
                    loss = output.mean()
                    loss.backward()
                    ddp_model.finish_gradient_synchronization()
                    optimizer.step()
                    optimizer.zero_grad()
                    prof.step()
        except Exception as e:
            print(f"Profiler warning: {e}")

    # 只有rank 0返回结果
    if global_rank == 0:
        result_data = {
            "config": {
                "use_bucket": use_bucket,
                "bucket_size_mb": bucket_size_mb if use_bucket else None,
                "world_size": world_size,
            },
            "ranks": all_results,
            "global_avg_time_ms": sum(r["avg_time_ms"] for r in all_results) / world_size,
        }

        # 通过queue返回
        if result_queue is not None:
            result_queue.put(result_data)

        # 同时写入文件（备用）
        if result_file is not None:
            with open(result_file, 'wb') as f:
                pickle.dump(result_data, f)

    dist.destroy_process_group()


def benchmark_worker(rank, world_size, bucket_size, use_bucket, num_iter, 
                     master_addr, master_port, result_queue, result_file):
    """包装函数用于spawn"""
    try:
        run_benchmark(rank, world_size, bucket_size, use_bucket, num_iter,
                      master_addr, master_port, result_queue, result_file)
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise


def run_single_test(world_size, bucket_size, use_bucket, num_iterations,
                   master_addr, master_port):
    """运行单次测试并返回结果"""

    with Manager() as manager:
        result_queue = manager.Queue()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            result_file = tmp.name

        try:
            spawn(
                benchmark_worker,
                args=(world_size, bucket_size, use_bucket, num_iterations,
                      master_addr, master_port, result_queue, result_file),
                nprocs=world_size,
                join=True
            )

            if not result_queue.empty():
                return result_queue.get()
            else:
                with open(result_file, 'rb') as f:
                    return pickle.load(f)
        finally:
            if os.path.exists(result_file):
                os.unlink(result_file)


def main():
    parser = argparse.ArgumentParser(description="DDP Bucketed Benchmark")
    parser.add_argument("--world_size", type=int, default=2, 
                       help="GPU数量")
    parser.add_argument("--num_iterations", type=int, default=20,
                       help="每次测试的迭代次数")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要CUDA"
    assert args.world_size <= torch.cuda.device_count(), "GPU数量不足"

    results = []
    bucket_sizes = [1, 10, 100, 1000]

    print("=" * 70)
    print("DDP Bucketed Benchmark")
    print(f"Configuration: 1 node, {args.world_size} GPUs, 0.5B model")
    print("=" * 70)

    # 1. 先测试无分桶版本（baseline）
    print("\n[1/5] Testing No Bucket (Baseline)...")
    result = run_single_test(
        args.world_size, 0, False, args.num_iterations,
        args.master_addr, args.master_port
    )
    results.append(result)

    # 2. 测试各种桶大小
    for i, bucket_size in enumerate(bucket_sizes, 2):
        print(f"\n[{i}/5] Testing Bucket Size = {bucket_size} MB...")
        result = run_single_test(
            args.world_size, bucket_size, True, args.num_iterations,
            args.master_addr, args.master_port
        )
        results.append(result)

    # 汇总结果
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    baseline = None
    bucketed_results = []

    for r in results:
        if not r["config"]["use_bucket"]:
            baseline = r
        else:
            bucketed_results.append(r)

    if baseline is None:
        print("ERROR: Baseline result not found!")
        return

    baseline_time = baseline["global_avg_time_ms"]
    print(f"\nBaseline (No Bucket): {baseline_time:.2f} ms/iteration")
    print("-" * 60)
    print(f"{'Config':<20} {'Time (ms)':<12} {'Overhead':<12} {'Speedup':<10}")
    print("-" * 60)

    print(f"{'No Bucket':<20} {baseline_time:<12.2f} {0.0:<12.2f} {1.0:<10.2f}x")

    for r in sorted(bucketed_results, key=lambda x: x["config"]["bucket_size_mb"]):
        bucket_size = r["config"]["bucket_size_mb"]
        time_ms = r["global_avg_time_ms"]
        overhead = time_ms - baseline_time
        speedup = baseline_time / time_ms if time_ms > 0 else float('inf')

        config_str = f"Bucket={bucket_size}MB"
        print(f"{config_str:<20} {time_ms:<12.2f} {overhead:<12.2f} {speedup:<10.2f}x")

    # 保存详细结果
    with open("benchmark_results.json", "w") as f:
        json_results = []
        for r in results:
            json_r = {
                "config": r["config"],
                "global_avg_time_ms": float(r["global_avg_time_ms"]),
                "ranks": [
                    {
                        "rank": rank_r["rank"],
                        "bucket_size_mb": float(rank_r["bucket_size_mb"]) if rank_r["bucket_size_mb"] != float('inf') else "inf",
                        "use_bucket": rank_r["use_bucket"],
                        "avg_time_ms": float(rank_r["avg_time_ms"]),
                        "min_time_ms": float(rank_r["min_time_ms"]),
                        "max_time_ms": float(rank_r["max_time_ms"]),
                    }
                    for rank_r in r["ranks"]
                ]
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    print(f"\n详细结果已保存到 benchmark_results.json")


if __name__ == "__main__":
    main()
