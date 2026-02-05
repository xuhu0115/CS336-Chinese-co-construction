from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.multiprocessing as mp

from .adapters import get_sharded_optimizer
from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # 使用CPU的gloo后端
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    # 由于我们已经设置了种子，模型状态在各个rank之间应该相同，而不需要广播。
    non_sharded_model = model_class().to(device)

    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(
        sharded_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()

        # 批次大小32，10个输入特征，5个输出特征
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        non_sharded_optimizer.step()
        sharded_optimizer.step()

    # 检查最终模型权重是否相同，无论我们使用的是
    # 分片优化器还是非分片优化器。
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )
    _cleanup_process_group()
