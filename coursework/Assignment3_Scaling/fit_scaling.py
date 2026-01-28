import numpy as np
from scaling import *


data = np.load("ckpt/scaling_records_70M.npy", allow_pickle=True)

flops = np.array([d["flops"] for d in data])
losses = np.array([d["loss"] for d in data])
D_data = np.array([d["tokens"] for d in data])  # 需要你有tokens记录

popt, pcov = fit_chinchilla_scaling(flops, D_data, losses)

plot_fixed_N(
    flops,
    losses,
    popt,
    6e7,
    save_path="ckpt/fix_N_70M.png"
)


plot_optimal_curve(
    popt,
    N_min=1e6,
    N_max=1e10,
    num_points=200,
    save_path="ckpt/optimal_curve_70M.png"
)