import torch

class FlashAttentionAutograd(torch.autograd.Function):
    """
    FlashAttention（forward + backward）

    输入：
        Q: (B, Nq, d)
        K: (B, Nk, d)
        V: (B, Nk, d)

    输出：
        O: (B, Nq, d)
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        B, Nq, d = Q.shape
        Nk = K.shape[1]

        # -----------------------------
        # block size
        # -----------------------------
        Bq = 64  # Query block 的尺寸
        Bk = 64  # Key / Value block 的尺寸

        # 我们将KV矩阵分成一个个小矩阵

        Tq = Nq // Bq  # Q block 数
        Tk = Nk // Bk  # K/V block 数

        scale = d ** -0.5

        # -----------------------------
        # 输出与中间量
        # -----------------------------
        O = torch.zeros_like(Q)              # Attention 输出
        L = torch.zeros(B, Nq, device=Q.device)  # log-sum-exp（给 backward 用）

        # =========================================================
        # 对 batch 维度显式循环
        # =========================================================
        for b in range(B):
            Q_b = Q[b]  # (Nq, d)
            K_b = K[b]  # (Nk, d)
            V_b = V[b]  # (Nk, d)

            # =====================================================
            # 外层循环：Query block
            # =====================================================
            for i in range(Tq):
                q_i = Q_b[i * Bq:(i + 1) * Bq]  # (Bq, d)

                # -----------------------------
                # FlashAttention 核心状态
                # -----------------------------
                m = torch.full((Bq, 1), -float("inf"), device=Q.device)  # running max
                l = torch.zeros((Bq, 1), device=Q.device)               # running sum
                O_acc = torch.zeros((Bq, d), device=Q.device)           # 输出累积（未归一化）

                # =================================================
                # 内层循环：Key / Value block
                # =================================================
                for j in range(Tk):
                    k_j = K_b[j * Bk:(j + 1) * Bk]  # (Bk, d)
                    v_j = V_b[j * Bk:(j + 1) * Bk]  # (Bk, d)

                    # --------------------------------------------
                    # 计算局部 attention score
                    # S_ij = Q_i @ K_j^T / sqrt(d)
                    # --------------------------------------------
                    S = q_i @ k_j.T * scale  # (Bq, Bk)

                    # 当前 block 内，每一行的最大值
                    row_max = S.max(dim=1, keepdim=True).values  # (Bq, 1)

                    # 更新 running max
                    m_new = torch.maximum(m, row_max)

                    # --------------------------------------------
                    # 数值稳定 softmax（FlashAttention 关键）
                    # --------------------------------------------
                    # 旧 block 贡献修正
                    exp_old = torch.exp(m - m_new) * l

                    # 当前 block 的 exp
                    P = torch.exp(S - m_new)

                    # 更新分母
                    l = exp_old + P.sum(dim=1, keepdim=True)

                    # 更新未归一化输出
                    O_acc = (
                        torch.exp(m - m_new) * O_acc +
                        P @ v_j
                    )

                    # 写回 running max
                    m = m_new

                # =================================================
                # block 内 softmax 归一化
                # =================================================
                O_i = O_acc / l  # (Bq, d)

                # log-sum-exp：L = m + log(l)
                L_i = m.squeeze(1) + torch.log(l.squeeze(1))

                # 写回全局输出
                O[b, i * Bq:(i + 1) * Bq] = O_i
                L[b, i * Bq:(i + 1) * Bq] = L_i

        # 保存 backward 所需变量
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.Bq = Bq
        ctx.Bk = Bk

        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        Bq = ctx.Bq
        Bk = ctx.Bk

        B, Nq, d = Q.shape
        Nk = K.shape[1]

        Tq = Nq // Bq
        Tk = Nk // Bk

        # --------------------------------------------
        # FlashAttention backward 中的重要中间量
        # D_i = sum_j O_ij * dO_ij
        # --------------------------------------------
        D = torch.sum(O * dO, dim=-1)  # (B, Nq)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for b in range(B):
            for i in range(Tq):
                q_i = Q[b, i * Bq:(i + 1) * Bq]
                dO_i = dO[b, i * Bq:(i + 1) * Bq]
                L_i = L[b, i * Bq:(i + 1) * Bq]
                D_i = D[b, i * Bq:(i + 1) * Bq]

                for j in range(Tk):
                    k_j = K[b, j * Bk:(j + 1) * Bk]
                    v_j = V[b, j * Bk:(j + 1) * Bk]

                    # --------------------------------------------
                    # 重算局部 attention 概率 P_ij
                    # P_ij = exp(S_ij - L_i)
                    # --------------------------------------------
                    S = q_i @ k_j.T * scale
                    P = torch.exp(S - L_i.unsqueeze(1))

                    # -------- dV --------
                    dV[b, j * Bk:(j + 1) * Bk] += P.T @ dO_i

                    # -------- dS --------
                    dP = dO_i @ v_j.T
                    dS = P * (dP - D_i.unsqueeze(1))

                    # -------- dQ --------
                    dQ[b, i * Bq:(i + 1) * Bq] += dS @ k_j * scale

                    # -------- dK --------
                    dK[b, j * Bk:(j + 1) * Bk] += dS.T @ q_i * scale

        return dQ, dK, dV
