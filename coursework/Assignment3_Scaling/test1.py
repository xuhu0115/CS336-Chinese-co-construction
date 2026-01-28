import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 加载数据
with open('./isoflops_curves.json', 'r') as f:
    data = json.load(f)

# 提取每个compute_budget下的最优N(使loss最小的点)
budgets = sorted(list(set(d['compute_budget'] for d in data)))
optimal_points = []

for c in budgets:
    subset = [d for d in data if d['compute_budget'] == c]
    best_entry = min(subset, key=lambda x: x['final_loss'])

    n_opt = best_entry['parameters']
    # 根据C = 6ND计算D(Tokens)
    d_opt = c / (6 * n_opt)

    optimal_points.append((c, d_opt))

# 转换为numpy数组便于计算
c_vals = np.array([p[0] for p in optimal_points])
d_vals = np.array([p[1] for p in optimal_points])


# 幂律拟合: D = k * C^G -> log(D) = log(k) + G * log(C)
def power_law(C, k, G):
    return k * np.power(C, G)

# 使用curve_fit进行拟合
params, covariance = curve_fit(power_law, c_vals, d_vals, p0=[1e-1, 0.5])
k_fit, G_fit = params

print(f"拟合公式: D_opt = {k_fit:.4e} * C^{G_fit:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(c_vals, d_vals, label='Derived points (C, Dopt)', color='tab:blue', zorder=3)

c_range = np.logspace(np.log10(c_vals.min()), np.log10(1e24), 100)
d_fit = power_law(c_range, k_fit, G_fit)
plt.loglog(c_range, d_fit, label=f'Fit: Dopt = k·C^{G_fit:.3f}', color='tab:blue')

# 预测图片中的点
for pred_c in [1e23, 1e24]:
    pred_d = power_law(pred_c, k_fit, G_fit)
    plt.scatter(pred_c, pred_d, marker='x', s=100)
    plt.text(pred_c, pred_d, f' {pred_c:.0e}: {pred_d:.2e}', verticalalignment='bottom')

# 设置坐标轴和样式
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Compute budget C (FLOPs)')
plt.ylabel('Optimal dataset size Dopt (tokens, proxy)')
plt.title('IsoFLOPs Scaling Law: Dataset Size vs Compute (Fitted from JSON)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()

save_path ='./Scaling Law.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"拟合完成！图像已自动保存至: {os.path.abspath(save_path)}")
print(f"拟合公式参数: k = {k_fit:.4e}, G = {G_fit:.4f}")
#plt.show()