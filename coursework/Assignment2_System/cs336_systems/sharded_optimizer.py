from typing import Any, Type, Iterable, Dict, List
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Optimizer State Sharding (ZeRO-1 style, simplified)

    Each rank owns a shard of parameters and maintains optimizer
    states only for those parameters. After each step, updated
    parameters are broadcast to all other ranks.
    """

    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # super() initializes param_groups and calls add_param_group
        super().__init__(params, defaults={})

        # Build the local (sharded) optimizer
        self._build_local_optimizer()

    def _build_local_optimizer(self):
        """
        Create the wrapped optimizer using only parameters
        assigned to this rank.
        """
        local_param_groups: List[Dict[str, Any]] = []

        for group in self.param_groups:
            local_params = [
                p for p in group["params"]
                if self._param_owner(p) == self.rank
            ]

            if len(local_params) == 0:
                continue

            local_group = dict(group)
            local_group["params"] = local_params
            local_param_groups.append(local_group)

        self.local_optimizer = self.optimizer_cls(
            local_param_groups,
            **self.optimizer_kwargs,
        )

    def _param_owner(self, param: torch.nn.Parameter) -> int:
        """
        Determine which rank owns this parameter.
        """
        return self._global_param_index(param) % self.world_size

    def _global_param_index(self, param: torch.nn.Parameter) -> int:
        """
        Assign a stable global index to each parameter.
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
        Run optimizer step on local shard, then synchronize parameters.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step only updates local shard
        self.local_optimizer.step(**kwargs)

        # Synchronize updated parameters
        self._sync_parameters()

        return loss

    def _sync_parameters(self):
        """
        Broadcast updated parameters from owning rank to all others.
        """
        for group in self.param_groups:
            for p in group["params"]:
                owner = self._param_owner(p)
                dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a parameter group and rebuild local optimizer.
        """
        super().add_param_group(param_group)
        self._build_local_optimizer()
