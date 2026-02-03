from typing import Any, Type, Iterable, Dict, List
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    分片优化器（Optimizer State Sharding，简化版 ZeRO-1）

    核心思想：
    - 所有 rank 共享完整的模型参数（参数本身不切分）
    - 但每个 rank 只“负责”其中一部分参数的优化器状态（如 Adam 的 m / v）
    - 每一步：
        1. 各 rank 只对自己负责的参数执行 optimizer.step()
        2. 然后通过 broadcast，把更新后的参数同步到所有 rank

    这样可以：
    - 将优化器状态显存占用降低到原来的 1 / world_size
    - 逻辑上等价于单卡训练（在同步完成后）
    """

    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        # ZeRO / 分片优化器必须运行在 torch.distributed 初始化之后
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        # 当前进程的 rank（全局唯一）
        self.rank = dist.get_rank()

        # 总进程数（world size）
        self.world_size = dist.get_world_size()

        # 被包装的真实优化器类型（如 torch.optim.Adam）
        self.optimizer_cls = optimizer_cls

        # 真实优化器的超参数（lr、betas 等）
        self.optimizer_kwargs = kwargs

        # 调用 Optimizer 的构造函数
        # 作用：
        # - 初始化 self.param_groups
        # - 为每个参数组调用 add_param_group
        super().__init__(params, defaults={})

        # 构建“本 rank 专属”的本地优化器
        self._build_local_optimizer()

    def _build_local_optimizer(self):
        """
        构建当前 rank 的本地优化器（local optimizer）

        核心逻辑：
        - 遍历所有参数
        - 只挑选“属于当前 rank”的参数
        - 用这些参数创建一个真正执行 step() 的 optimizer

        注意：
        - 每个 rank 的 local_optimizer 看到的参数子集不同
        - 但 param_groups 的结构（lr / weight_decay 等）保持一致
        """
        local_param_groups: List[Dict[str, Any]] = []

        for group in self.param_groups:
            # 从当前参数组中，筛选出 owner == 当前 rank 的参数
            local_params = [
                p for p in group["params"]
                if self._param_owner(p) == self.rank
            ]

            # 如果该参数组在当前 rank 没有任何参数，直接跳过
            if len(local_params) == 0:
                continue

            # 复制一份参数组配置（避免修改原始 param_groups）
            local_group = dict(group)

            # 用本 rank 负责的参数替换 params
            local_group["params"] = local_params

            local_param_groups.append(local_group)

        # 用筛选后的参数组，实例化真实的优化器（如 Adam）
        self.local_optimizer = self.optimizer_cls(
            local_param_groups,
            **self.optimizer_kwargs,
        )

    def _param_owner(self, param: torch.nn.Parameter) -> int:
        """
        判断某个参数“归属”哪个 rank

        当前实现策略：
        - 先为每个参数分配一个全局唯一的 index
        - owner = index % world_size

        这是最简单、最常见的静态分片方式
        """
        return self._global_param_index(param) % self.world_size

    def _global_param_index(self, param: torch.nn.Parameter) -> int:
        """
        为每一个参数分配一个稳定的“全局索引”

        设计要求：
        - 所有 rank 上，参数遍历顺序必须一致
        - 同一个参数在不同 rank 上，index 必须完全相同

        实现方式：
        - 第一次调用时，遍历所有 param_groups
        - 按出现顺序给每个 parameter 编号
        - 使用字典缓存 param -> index 的映射
        """
        if not hasattr(self, "_param_to_index"):
            self._param_to_index = {}
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    self._param_to_index[p] = idx
                    idx += 1
        return self._param_to_index[param]

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        执行一次优化步骤（等价于 Optimizer.step）

        流程：
        1. （可选）执行 closure 计算 loss
        2. 仅对当前 rank 负责的参数执行 optimizer.step()
        3. 将更新后的参数广播给所有 rank，同步模型参数

        注意：
        - 优化器状态（如 Adam 的动量）始终只存在于 owner rank
        - 参数本身在 step 结束后在所有 rank 上保持一致
        """
        loss = None
        if closure is not None:
            # closure 需要开启梯度
            with torch.enable_grad():
                loss = closure()

        #  这里只会更新“本 rank 拥有的参数”
        self.local_optimizer.step(**kwargs)

        # 同步参数：owner rank → 其他所有 rank
        self._sync_parameters()

        return loss

    def _sync_parameters(self):
        """
        参数同步逻辑（broadcast）

        对每一个参数：
        - 找到它的 owner rank
        - 从 owner rank 将参数广播到所有 rank

        这样可以保证：
        - 虽然每个参数只在一个 rank 上被更新
        - 但 step 结束后，所有 rank 上的参数值完全一致
        """
        for group in self.param_groups:
            for p in group["params"]:
                owner = self._param_owner(p)
                dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        动态添加参数组（与 PyTorch Optimizer 接口保持一致）

        由于参数发生变化：
        - 需要重新计算参数分片
        - 并重建 local optimizer
        """
        super().add_param_group(param_group)
        self._build_local_optimizer()
