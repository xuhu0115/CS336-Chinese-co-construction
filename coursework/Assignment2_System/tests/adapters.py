from __future__ import annotations

from typing import Type

import torch



def get_flashattention_autograd_function_pytorch() -> Type:
    """
    返回一个实现FlashAttention2的torch.autograd.Function子类。
    期望这个类使用标准PyTorch操作（不使用Triton！）来实现FlashAttention2。

    Returns:
        一个类对象（不是类的实例）
    """
    # 例如: return MyFlashAttnAutogradFunctionClass
    raise NotImplementedError


def get_flashattention_autograd_function_triton() -> Type:
    """
    返回一个使用Triton内核实现FlashAttention2的torch.autograd.Function子类。
    期望这个类实现与get_flashattention_autograd_function_pytorch()中返回的类相同的操作，
    但它应该通过在前向和后向传递中调用自定义Triton内核来实现。

    Returns:
        一个类对象（不是类的实例）
    """
    # 例如: return MyTritonFlashAttentionAutogradFunctionClass
    raise NotImplementedError


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    返回一个处理分布式数据并行训练中参数广播和梯度同步的torch.nn.Module容器。

    该容器应该通过在反向传播中异步传递就绪的梯度来重叠通信与反向传播计算。
    每个参数张量的梯度单独通信。

    Args:
        module: torch.nn.Module
            要用DDP包装的底层模型。
    Returns:
        DDP类的实例。
    """
    # 例如: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在完成反向传播之后，但在执行优化器步骤之前运行的代码。

    Args:
        ddp_model: torch.nn.Module
            DDP包装的模型。
        optimizer: torch.optim.Optimizer
            与DDP包装模型一起使用的优化器。
    """
    # 例如: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    返回一个处理分布式数据并行训练中参数广播和梯度同步的torch.nn.Module容器。

    该容器应该通过在反向传播中异步传递就绪的梯度桶来重叠通信与反向传播计算。

    Args:
        module: torch.nn.Module
            要用DDP包装的底层模型。
        bucket_size_mb: 桶大小，以兆字节为单位。如果为None，使用单个无界大小的桶。
    Returns:
        DDP类的实例。
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在完成反向传播之后，但在执行优化器步骤之前运行的代码。

    Args:
        ddp_model: torch.nn.Module
            DDP包装的模型。
        optimizer: torch.optim.Optimizer
            与DDP包装模型一起使用的优化器。
    """
    # 例如: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在训练步骤最开始时运行的代码。

    Args:
        ddp_model: torch.nn.Module
            DDP包装的模型。
        optimizer: torch.optim.Optimizer
            与DDP包装模型一起使用的优化器。
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    返回一个处理给定优化器类在提供的参数上优化器状态分片的torch.optim.Optimizer。

    Arguments:
        params (``Iterable``): 一个包含所有参数的``Iterable``，这些参数将是:class:`torch.Tensor` s
            或:class:`dict` s，将在各rank之间分片。
        optimizer_class (:class:`torch.nn.Optimizer`): 本地
            优化器的类。
    Keyword arguments:
        kwargs: 要转发给优化器构造器的关键字参数。
    Returns:
        分片优化器的实例。
    """
    raise NotImplementedError
