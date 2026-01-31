import logging
from copy import deepcopy
from typing import Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .adapters import (
    ddp_individual_parameters_on_after_backward,
    get_ddp_individual_parameters,
)
from .common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_DistributedDataParallelIndividualParameters(model_class):
    world_size = 2
    mp.spawn(
        _test_DistributedDataParallelIndividualParameters,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_DistributedDataParallelIndividualParameters(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # 使用CPU的gloo后端
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # 在运行测试前执行barrier以确保每个进程
    # 已经完成初始化，并且以下测试
    # 由于跳过而立即退出不会导致不稳定。
    dist.barrier()

    # 设置种子以确保rank使用不同的初始模型进行初始化。
    torch.manual_seed(rank)

    # 创建一个玩具模型并将其移动到适当的设备。
    # 这是我们的非并行基线。
    non_parallel_model = model_class().to(device)

    # 创建一个DDP模型。请注意，此模型的权重应该
    # 与上面的非并行基线匹配。
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_individual_parameters(ddp_base)

    # 如果我们在rank 0上，DDP模型仍然应该与
    # 非并行基线的参数完全匹配（因为rank 0上的参数没有改变）。
    # 如果我们不在rank 0上，DDP模型的参数应该已经用
    # rank 0的参数进行了更新。因此，再次检查参数是否与
    # 局部初始状态不同。
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # 此参数初始化为[2, 2]，所以我们期望其值保持不变
        is_no_grad_fixed_param = (
            "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        )
        if rank == 0 or is_no_grad_fixed_param:
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # 确保所有rank具有相同的模型状态
    validate_ddp_net_equivalence(ddp_model)

    # 从磁盘加载数据集，以便我们可以确保每个rank具有相同的
    # 整体数据池。
    # 形状: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # 形状: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    # 每个rank只会看到10个样本（总共20个样本的数据集大小）
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # DDP模型的优化器
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # 非并行模型的优化器
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(5):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # 在所有数据上运行非并行模型并执行梯度步骤
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # 此时，非并行模型的参数应该与
        # DDP模型的参数不同（因为我们已经对
        # 非并行模型应用了梯度步骤，但没有对DDP模型应用）。
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # 唯一会改变的参数是那些requires_grad的参数
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # 不需要requires_grad的参数不应该改变
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # 虽然非并行模型对所有数据（20个样本）进行前向传播，
        # 每个DDP rank只看到10个（不相交的）样本。
        # 但是，最终结果应该与对所有20个样本进行前向传播相同。
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # 运行学生编写的代码，需要在反向传播之后、
        # 优化器步骤之前执行（例如，等待所有DDP rank同步梯度）
        ddp_individual_parameters_on_after_backward(ddp_model, ddp_optimizer)

        ddp_optimizer.step()

        # 此时，非并行模型应该与DDP模型的参数完全匹配
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # 打乱数据，以便在下一次迭代中，每个DDP rank看到不同的输入集合。
        # 我们确保在打乱时使用相同的种子（否则每个rank的样本可能不是不相交的）。
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # 训练结束后，我们应该在非并行基线和
    # 用DDP训练的模型上获得相同的权重。
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
    _cleanup_process_group()
