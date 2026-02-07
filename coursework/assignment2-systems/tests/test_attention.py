import pytest
import torch
from einops import einsum, rearrange

from .adapters import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)


def _attention_and_lse(q, k, v, is_causal=False):
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d ** 0.5)
    S = einsum(q, k, '... q d, ... k d -> ... q k') * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
            S,
            -1e6
        )
    P = torch.softmax(S, dim=-1)
    o = einsum(P, v, '... q k, ... k d -> ... q d')
    L = torch.logsumexp(S, dim=-1)
    return o, L


def _make_attn_inputs(device=None):
    torch.random.manual_seed(0)
    batch_size = 4
    n_queries = 128
    n_keys = 128
    D = 64
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_queries, D, device=device)

    return q, k, v, do


def _test_flash_forward_pass(impl, device="cpu", is_causal=False):
    q, k, v, _do = _make_attn_inputs(device)
    o = impl(q, k, v, is_causal)

    # Extract L from the saved tensors
    assert o.grad_fn.saved_tensors is not None, "No saved tensors found in the output tensor. Make sure your autograd forward is saving them using ctx.save_for_backward."
    maybe_ls = [t for t in o.grad_fn.saved_tensors if t.shape == (q.shape[0], q.shape[1])]

    assert len(maybe_ls) == 1, f"Expected one tensor of shape {q.shape[0], q.shape[1]} in saved tensors, but found {len(maybe_ls)}. The tests require you to save exactly one tensor of this shape, corresponding to the log-sum-exp of the attention scores."
    l = maybe_ls[0]

    o_ref, l_ref = _attention_and_lse(q, k, v, is_causal)

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(l, l_ref, rtol=1e-2, atol=1e-2)


def test_flash_forward_pass_pytorch():
    _test_flash_forward_pass(get_flashattention_autograd_function_pytorch().apply)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_forward_pass_triton(is_causal):
    _test_flash_forward_pass(get_flashattention_autograd_function_triton().apply, device="cuda", is_causal=is_causal)



def flash_backward_results(impl, is_causal, device=None):
    q, k, v, do = _make_attn_inputs(device=device)
    impl(q, k, v, is_causal).backward(do) 
    return q.grad, k.grad, v.grad


def test_flash_backward_pytorch():
    dq_expected, dk_expected, dv_expected = flash_backward_results(lambda *args: _attention_and_lse(*args)[0], False)

    q, k, v, do = _make_attn_inputs()
    get_flashattention_autograd_function_pytorch().apply(q, k, v, False).backward(do)

    torch.testing.assert_close(dq_expected, q.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dk_expected, k.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dv_expected, v.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_backward_triton(is_causal):
    dq_expected, dk_expected, dv_expected = flash_backward_results(lambda *args: _attention_and_lse(*args)[0], is_causal, device='cuda')

    q, k, v, do = _make_attn_inputs(device='cuda')
    get_flashattention_autograd_function_triton().apply(q, k, v, is_causal).backward(do)

    torch.testing.assert_close(dq_expected, q.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dk_expected, k.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dv_expected, v.grad, rtol=1e-2, atol=1e-2)