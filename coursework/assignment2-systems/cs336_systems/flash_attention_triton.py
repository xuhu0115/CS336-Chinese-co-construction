# from sympy import Q
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd_kernel(
    # -------- 全局内存指针 --------
    Q_ptr, K_ptr, V_ptr,           # 输入 Q, K, V
    O_ptr, L_ptr,                  # 输出 O, logsumexp L

    # -------- stride 信息（用于 block ptr）--------
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,

    # -------- 序列长度 --------
    Nq, Nk,

    # -------- scale --------
    scale,

    # -------- 编译期常量 --------
    D: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    每个 Triton program 负责：
        - 一个 batch
        - 一个 Query block（Q_BLOCK 行）
    """

    # ------------------------------------------------
    # Triton 并行索引
    # ------------------------------------------------
    q_block_id = tl.program_id(0)     # 第几个 Q block
    batch_id   = tl.program_id(1)     # 第几个 batch

    # =================================================
    # 构造 block pointer（这是 Triton 的关键）
    # =================================================
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb,
        shape=(Nq, D),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    # K / V 从第 0 行开始，后面在 loop 中 advance
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_id * stride_kb,
        shape=(Nk, D),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_id * stride_vb,
        shape=(Nk, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_id * stride_ob,
        shape=(Nq, D),
        strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_id * stride_lb,
        shape=(Nq, 1),
        strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    # =================================================
    # 加载 Query block
    # =================================================
    Q_i = tl.load(Q_block_ptr)  # (Q_BLOCK, D)

    # =================================================
    # FlashAttention 核心状态（每个 Q block 独立）
    # =================================================
    O_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)       # 未归一化输出累积
    L_acc = tl.zeros((Q_BLOCK, 1), dtype=tl.float32)      # softmax 分母
    M_acc = tl.full((Q_BLOCK, 1), -float("inf"), tl.float32)  # running max

    # =================================================
    # 内层循环：遍历所有 K/V block
    # =================================================
    for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
        K_j = tl.load(K_block_ptr)   # (K_BLOCK, D)
        V_j = tl.load(V_block_ptr)   # (K_BLOCK, D)

        # ---------------------------------------------
        # S_ij = Q_i @ K_j^T / sqrt(d)
        # ---------------------------------------------
        S_ij = tl.dot(Q_i, K_j.T) * scale  # (Q_BLOCK, K_BLOCK)

        # ---------------------------------------------
        # Causal mask（编译期 if）
        # ---------------------------------------------
        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # ---------------------------------------------
        # 数值稳定 softmax（FlashAttention 核心）
        # ---------------------------------------------
        M_block = tl.max(S_ij, axis=1, keep_dims=True)
        M_new = tl.maximum(M_acc, M_block)

        P_ij = tl.exp(S_ij - M_block)

        L_new = (
            tl.exp(M_acc - M_new) * L_acc +
            tl.exp(M_block - M_new) * tl.sum(P_ij, axis=1, keep_dims=True)
        )

        # 类型对齐（Triton 细节）
        P_cast = P_ij.to(V_block_ptr.type.element_ty)

        O_new = (
            tl.exp(M_acc - M_new) * O_acc +
            tl.exp(M_block - M_new) * tl.dot(P_cast, V_j)
        )

        # ---------------------------------------------
        # 更新 running 状态
        # ---------------------------------------------
        M_acc = M_new
        L_acc = L_new
        O_acc = O_new

        # ---------------------------------------------
        # 移动 K/V block 指针
        # ---------------------------------------------
        K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK, 0))

    # =================================================
    # softmax 归一化
    # =================================================
    O_i = O_acc / L_acc
    L_i = M_acc + tl.log(L_acc)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (B, N, D)
        """

        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q_BLOCK = 64
        K_BLOCK = 64

        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device)

        grid = (Nq // Q_BLOCK, B)

        flash_attention_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, L)
        ctx.is_causal = is_causal
        return O
