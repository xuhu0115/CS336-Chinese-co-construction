import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. FLOPs估计
def estimate_flops(num_params, tokens_seen):
    # FLOPs ≈ 6 * N * D
    return 6.0 * num_params * tokens_seen

# DeepMind Chinchilla Scaling Law
def chinchilla_scaling_law(inputs, E, A, B, alpha, beta):
    """
    inputs: tuple (N, D)
        N: 模型参数量
        D: 训练Token量
    """
    N, D = inputs
    return E + A * (N ** -alpha) + B * (D ** -beta)


def fit_chinchilla_scaling(N_array, D_array, losses):
    # E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
    p0 = [1.69, 400.0, 400.0, 0.33, 0.28]

    # 边界约束
    bounds = (
        [0.0, 0.0, 0.0, 0.0, 0.0],  # 下界
        [10.0, 1e5, 1e5, 1.0, 1.0]  # 上界
    )

    # popt = [E, A, B, alpha, beta]
    # curve_fit()是用来根据实验数据得到相应的最优的参数解
    popt, pcov = curve_fit(
        chinchilla_scaling_law,
        (N_array, D_array),  # 输入是元组
        losses,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )
    return popt, pcov


# 固定模型规模N，看FLOPs-loss的关系
def plot_fixed_N(
    flops,
    losses,
    popt,
    N_fixed,
    save_path=None
):
    flops = np.array(flops)
    losses = np.array(losses)

    flops_fit = np.logspace(
        np.log10(flops.min()),
        np.log10(flops.max()),
        200
    )

    # FLOPs ≈ 6 * N_{no_embedd} * D_{token}
    D_fit = flops_fit / (6.0 * N_fixed)

    # N_fixed值得有效学习参数（非嵌入参数）
    N_fit = np.full_like(D_fit, N_fixed)

    losses_fit = chinchilla_scaling_law(
        (N_fit, D_fit),
        *popt
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(flops, losses, s=25, label="Observed")
    plt.plot(
        flops_fit,
        losses_fit,
        linewidth=2,
        label=f"Fitted (N={N_fixed:.1e})"
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training FLOPs (log)")
    plt.ylabel("Loss (log)")
    plt.title("Scaling (Fixed Model Size)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

# 在每一个FLOPs预算下找可以得到最小loss的(N, D)，拟合得到loss-FLOPs最优曲线
def plot_optimal_curve(
    popt,
    N_min=1e6,
    N_max=1e10,
    num_points=200,
    save_path=None
):
    E, A, B, alpha, beta = popt

    N = np.logspace(np.log10(N_min), np.log10(N_max), num_points)

    # Chinchilla optimal D(N)
    D = ((A * alpha) / (B * beta)) ** (1 / beta) * N ** (alpha / beta)

    FLOPs = 6.0 * N * D
    loss = chinchilla_scaling_law((N, D), *popt)

    plt.figure(figsize=(6, 5))
    plt.plot(FLOPs, loss, linewidth=2, label="Optimal Curve")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training FLOPs (log)")
    plt.ylabel("Loss (log)")
    plt.title("Optimal Compute")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

