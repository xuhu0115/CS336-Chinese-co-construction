import argparse
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import psutil  # 需要pip install psutil
import torch
from not_RMSNorm_model import *
# 训练前环境初始化（防止OMP冲突）
import os
from transformers import PreTrainedTokenizerFast

# 允许重复的OpenMP runtime（防止libiomp5md.dll冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 限制OpenMP线程数，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 可选：显示当前线程设置，便于调试
#print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
#print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")

class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None: continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Correct bias correction logic
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)
                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss


def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_out * (alpha_max - alpha_min)
    else:
        return alpha_min


def run_gradient_clipping(params, max_norm, eps=1e-6):
    params_with_grad = [p for p in params if p.grad is not None]
    if len(params_with_grad) == 0: return
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2)
    clip_coeff = max_norm / (total_norm + eps)
    if clip_coeff < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)


# Part 4: 数据加载
class CausalMemmapDataset(Dataset):
    def __init__(self, data_path, context_length, start_block=0, end_block=None):

        # 确保dtype一致，通常语料索引使用int32足够
        self.data = np.memmap(data_path, mode='r', dtype=np.int32)
        self.context_length = context_length

        total_blocks = (len(self.data) - context_length - 1) // context_length

        if end_block is None:
            end_block = total_blocks

        self.start_block = start_block
        self.end_block = end_block
        self.num_blocks = end_block - start_block

        # 简单的边界检查
        if self.num_blocks <= 0:
            print(f"Warning: Dataset has 0 blocks. (Start: {start_block}, End: {end_block})")

    def __len__(self):
        return max(0, self.num_blocks)

    def __getitem__(self, idx):
        block_idx = self.start_block + idx
        start_idx = block_idx * self.context_length

        # 转换为int64给torch使用
        x = torch.from_numpy(
            self.data[start_idx: start_idx + self.context_length].astype(np.int64)
        )

        y = torch.from_numpy(
            self.data[start_idx + 1: start_idx + self.context_length + 1].astype(np.int64)
        )

        return x, y


def save_ppl_curve(train_ppls, val_ppls, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Step-level PPL
    plt.figure()
    plt.plot(train_ppls)
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexity (per step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_ppl1.png"))
    plt.close()

    # Validation PPL
    plt.figure()
    plt.plot(val_ppls)
    plt.yscale("log")
    plt.xlabel("Val Step")
    plt.ylabel("Perplexity")
    plt.title("Val Perplexity (per step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_ppl1.png"))
    plt.close()


# Part 5: 训练循环
def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    epoch,
    config: dict
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "config": config,
    }

    torch.save(ckpt, path)

def get_memory_usage(device):
    """获取当前内存/显存占用情况"""
    if device == "cuda":
        # 获取当前设备已分配的显存(MB)
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        return f"{mem:.2f} MB (GPU)"
    else:
        # 获取当前进程占用的系统内存 (MB)
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        return f"{mem:.2f} MB (CPU)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="./Assignment1_Ablations/no_RMSNorm")
    parser.add_argument("--data_path", type=str, default="data.in")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 生成模拟数据 (dtype必须为int32以匹配Dataset读取)
    if not os.path.exists(args.data_path):
        print(f"创建模拟数据{args.data_path}...")
        # 保证有足够的数据生成若干batch
        dummy_len = args.context_length * args.batch_size * 20
        dummy_data = np.random.randint(0, args.vocab_size, (dummy_len,), dtype=np.int32)
        dummy_data.tofile(args.data_path)

    total_ds = CausalMemmapDataset(args.data_path, args.context_length)
    total_blocks = len(total_ds)
    split = 0.8
    split_block = int(total_blocks * split)

    train_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=0,
        end_block=split_block
    )

    val_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=split_block,
        end_block=total_blocks
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="./tokenizer.json"
    )

    vocab_size = tokenizer.vocab_size

    # 确保dataset不为空
    if len(train_ds) == 0:
        raise ValueError("训练数据集为空，增加数据量或减小上下文长度。")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # 统计容器
    train_ppls = []
    val_ppls = []

    # Epoch级别的统计
    epoch_avg_tlosses = []
    epoch_avg_tppls = []
    epoch_avg_vlosses = []
    epoch_avg_vppls = []

    # 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = CustomAdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    step_t = 0
    step_v = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()  # 记录开始时间
        model.train()  # 开启训练模式

        current_epoch_losses = []  # 用于计算当前Epoch平均Loss
        print(f"\n--- Epoch {epoch} ---")
        print(f"初始内存占用: {get_memory_usage(device)}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # 更新学习率
            lr = get_lr_cosine_schedule(step_t, args.lr, args.min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            logits = model(x)  # (B, T, Vocab)

            loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))

            # 变量名定义
            loss_val = loss.item()
            ppl_val = math.exp(loss_val)

            if not torch.isfinite(loss):
                # print("NaN loss, skip step")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            run_gradient_clipping(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录Step级别
            current_epoch_losses.append(loss_val)
            train_ppls.append(ppl_val)
            step_t += 1

            if batch_idx % 100 == 0:  # 减少打印频率
                mem_status = get_memory_usage(device)
                print(f"Step {batch_idx} | 实时内存: {mem_status}")
                print(f"Epoch {epoch} | Step {step_t}/{total_steps} | "
                      f"LR: {lr:.6f} | Train Loss: {loss_val:.4f} | Train PPL: {ppl_val:.4f}")

        # Epoch结束统计
        if len(current_epoch_losses) > 0:
            epoch_tloss = sum(current_epoch_losses) / len(current_epoch_losses)
            epoch_tppl = math.exp(epoch_tloss)
        else:
            epoch_tloss, epoch_tppl = 0.0, 0.0
        epoch_train_end_time = time.time()
        train_duration = epoch_train_end_time - epoch_start_time

        # 计算并打印训练统计
        epoch_tloss = sum(current_epoch_losses) / len(current_epoch_losses) if current_epoch_losses else 0
        print(f"[Epoch {epoch} 训练完成] 耗时: {train_duration:.2f}s | "
              f"平均Loss: {epoch_tloss:.4f} | 内存占用: {get_memory_usage(device)}")
        epoch_avg_tlosses.append(epoch_tloss)
        epoch_avg_tppls.append(epoch_tppl)

        if len(epoch_avg_tlosses) > 0:
            print(f"[Epoch {epoch} END] Train Avg Loss: {epoch_tloss:.4f} | Train PPL: {epoch_tppl:.2f}\n")

        # 验证
        val_start_time = time.time()
        model.eval()
        val_losses = []  # 初始化列表

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))
                val_losses.append(loss.item())
                step_v += 1
                val_ppl = math.exp(loss.item())

                val_ppls.append(val_ppl)

            if len(val_losses) > 0:
                epoch_vloss = sum(val_losses) / len(val_losses)
                epoch_vppl = math.exp(epoch_vloss)
            else:
                epoch_vloss, epoch_vppl = 0.0, 0.0

            epoch_avg_vlosses.append(epoch_vloss)
            epoch_avg_vppls.append(epoch_vppl)

            val_duration = time.time() - val_start_time
            epoch_vloss = sum(val_losses) / len(val_losses) if val_losses else 0
            print(f"[Epoch {epoch} 验证完成] 耗时: {val_duration:.2f}s | "
                  f"验证Loss: {epoch_vloss:.4f} | 内存占用: {get_memory_usage(device)}")

            print(f"[Epoch {epoch} END] Val Avg Loss: {epoch_vloss:.4f} | Val PPL: {epoch_vppl:.2f}")

        save_ppl_curve(
            train_ppls,
            val_ppls,
            args.checkpoint_dir
        )

        if epoch % 10 == 0:

            save_checkpoint(
                path=f"./Assignment1_Ablations/no_RMSNorm/epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                iteration=step_t,
                epoch=epoch,
                config={
                    "vocab_size": vocab_size,
                    "context_length": args.context_length,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "d_model": args.d_model,
                }
            )

if __name__ == "__main__":
    main()