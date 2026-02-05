import torch
import torch.distributed as dist


# ============================================================
# Part 1. 逐参数 Overlap 版本（最直观的 DDP 教学实现）
# ============================================================

class TeachingDDPOverlapPerParam(torch.nn.Module):
    """
    教学版 DDP（逐参数通信）：

    - 每个参数在 backward 完成后，立刻触发 async all-reduce
    - 实现计算（backward）与通信的 overlap
    - 逻辑直观，但通信粒度太细（工业中不用）
    """

    def __init__(self, model: torch.nn.Module):
        # ⚠️ 必须最先初始化父类
        super().__init__()

        self.model = model
        self.world_size = dist.get_world_size()

        # 保存所有 async all-reduce 的句柄
        self.async_handles = []

        # ----------------------------------------------------
        # Step 1. 初始化阶段：同步模型参数 & buffer
        # ----------------------------------------------------
        # 等价于 PyTorch DDP 的 initial broadcast
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)

            for buffer in self.model.buffers():
                dist.broadcast(buffer.data, src=0)

        # ----------------------------------------------------
        # Step 2. 注册 backward hook（核心）
        # ----------------------------------------------------
        # 当某个参数的梯度刚算完，就立刻通信
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._create_grad_hook()
                )

    def _create_grad_hook(self):
        """
        创建一个 backward hook（closure）

        hook 的触发时机：
        - 当前参数的梯度已经计算完成
        - 但整个 backward 还没结束
        """

        def hook_fn(param: torch.nn.Parameter):
            # 有些参数可能没有梯度（如被冻结）
            if param.grad is None:
                return

            # ------------------------------------------------
            # 对该参数的梯度做 async all-reduce
            # ------------------------------------------------
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

            # 保存句柄，step 前需要 wait
            self.async_handles.append(handle)

            # ⚠️ register_post_accumulate_grad_hook
            #    不允许返回任何值

        return hook_fn

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finalize_gradients(self):
        """
        在 optimizer.step() 之前调用：

        - 等待所有 async all-reduce 完成
        - 对梯度做平均
        """

        # 1. 等待所有通信完成
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()

        # 2. 梯度平均
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)


# ============================================================
# Part 2. Bucket 类：DDP 的核心数据结构
# ============================================================

class GradientBucket:
    """
    一个 bucket 管理一组参数的梯度通信：

    - 多个参数的梯度被 flatten 到一个连续 buffer
    - 当 bucket 中所有参数梯度 ready 后，触发一次 all-reduce
    """

    def __init__(self, params: list[torch.nn.Parameter], bucket_id: int):
        self.bucket_id = bucket_id
        self.params = params

        # 已经 ready 的参数数量
        self.ready_count = 0

        # async all-reduce 的通信句柄
        self.comm_handle = None

        # ----------------------------------------------------
        # Step 1. 创建连续的梯度 buffer
        # ----------------------------------------------------
        total_numel = sum(p.numel() for p in params)
        dtype = params[0].dtype
        device = params[0].device

        self.flat_buffer = torch.zeros(
            total_numel, dtype=dtype, device=device
        )

        # ----------------------------------------------------
        # Step 2. 为每个参数创建 buffer 视图
        # ----------------------------------------------------
        self.param_to_view = {}
        offset = 0
        for p in params:
            numel = p.numel()
            self.param_to_view[p] = (
                self.flat_buffer[offset: offset + numel]
                .view(p.shape)
            )
            offset += numel

    def reset(self):
        """
        在每个 forward 之前调用：
        - 清空计数器
        - 清空通信句柄
        """
        self.ready_count = 0
        self.comm_handle = None


# ============================================================
# Part 3. Bucket 化 DDP（接近 PyTorch DDP 真正实现）
# ============================================================

class TeachingDDPBucketed(torch.nn.Module):
    """
    教学版 Bucket DDP（工业级思路）：

    - 参数按 bucket 分组
    - 每个 bucket 一次 all-reduce
    - 在 backward 中实现计算 / 通信 overlap
    """

    def __init__(self, model: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.model = model
        self.world_size = dist.get_world_size()

        # bucket 大小（MB -> Bytes）
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024

        # ----------------------------------------------------
        # Step 1. 初始化参数同步
        # ----------------------------------------------------
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.model.buffers():
                dist.broadcast(buffer.data, src=0)

        # ----------------------------------------------------
        # Step 2. 参数分桶（非常关键）
        # ----------------------------------------------------
        self.buckets: list[GradientBucket] = []
        self.param_to_bucket: dict[torch.nn.Parameter, GradientBucket] = {}

        # 只考虑需要梯度的参数
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        # ⚠️ 反向传播顺序：从输出层到输入层
        # 所以 bucket 要按“反向顺序”构建
        reversed_params = list(reversed(trainable_params))

        current_bucket_params = []
        current_bucket_bytes = 0

        for param in reversed_params:
            param_bytes = param.numel() * param.element_size()

            # 如果当前 bucket 已满，先创建 bucket
            if (
                current_bucket_params
                and current_bucket_bytes + param_bytes > self.bucket_size_bytes
            ):
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_bytes = 0

            current_bucket_params.append(param)
            current_bucket_bytes += param_bytes

        # 最后一个 bucket
        if current_bucket_params:
            self._create_bucket(current_bucket_params)

        # ----------------------------------------------------
        # Step 3. 注册 backward hooks
        # ----------------------------------------------------
        for param in trainable_params:
            param.register_post_accumulate_grad_hook(
                self._create_bucket_hook(param)
            )

    def _create_bucket(self, params: list[torch.nn.Parameter]):
        bucket = GradientBucket(params, bucket_id=len(self.buckets))
        self.buckets.append(bucket)

        for p in params:
            self.param_to_bucket[p] = bucket

    def _create_bucket_hook(self, param: torch.nn.Parameter):
        """
        每个参数的 hook：
        - 把梯度拷贝进 bucket buffer
        - 如果 bucket 满了，立刻发起 async all-reduce
        """

        def hook_fn(p: torch.nn.Parameter):
            if p.grad is None:
                return

            bucket = self.param_to_bucket[p]

            # A. 拷贝梯度到 bucket buffer
            bucket.param_to_view[p].copy_(p.grad)

            # B. 标记一个参数 ready
            bucket.ready_count += 1

            # C. bucket 内所有参数 ready → 启动通信
            if bucket.ready_count == len(bucket.params):
                bucket.comm_handle = dist.all_reduce(
                    bucket.flat_buffer,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )

        return hook_fn

    def forward(self, *args, **kwargs):
        # 每次 forward 前重置 bucket 状态
        for bucket in self.buckets:
            bucket.reset()
        return self.model(*args, **kwargs)

    def finalize_gradients(self):
        """
        在 optimizer.step() 之前调用：

        - 等待所有 bucket 的通信完成
        - 对 bucket buffer 做平均
        - 拷贝回各参数的 p.grad
        """

        for bucket in self.buckets:
            # 1. 等待通信
            if bucket.comm_handle is not None:
                bucket.comm_handle.wait()

            # 2. 平均梯度
            bucket.flat_buffer.div_(self.world_size)

            # 3. 拷贝回参数梯度
            for p in bucket.params:
                if p.grad is not None:
                    p.grad.copy_(bucket.param_to_view[p])
