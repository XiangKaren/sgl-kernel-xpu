import pytest
import torch
from sgl_kernel import fp8_w8a16_gemm

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128

# (N, K) of the dense projections of Qwen3.6-35B-A3B at TP=2: full-attention
# qkv / o_proj, linear-attention in_proj_qkvz / out_proj, shared-expert down.
SHAPES = [(4608, 2048), (2048, 2048), (6144, 2048), (512, 2048), (2048, 256)]
M_VALUES = [1, 3, 16, 128, 1000]


def _quant_per_tensor(w: torch.Tensor):
    scale = (w.abs().max().float() / FP8_E4M3_MAX).clamp(min=1e-12)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    return w_fp8, scale.reshape(1), w_fp8.float() * scale


def _quant_block(w: torch.Tensor):
    n, k = w.shape
    n_blocks = (n + BLOCK - 1) // BLOCK
    padded = torch.zeros(n_blocks * BLOCK, k, dtype=torch.float32, device=w.device)
    padded[:n] = w.float()
    blocks = padded.view(n_blocks, BLOCK, k // BLOCK, BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX).clamp(min=1e-12)
    expanded = scale[:, None, :, None]
    w_fp8 = (blocks / expanded).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    w_dq = (w_fp8.float() * expanded).view(n_blocks * BLOCK, k)[:n]
    w_fp8 = w_fp8.view(n_blocks * BLOCK, k)[:n].contiguous()
    return w_fp8, scale.contiguous(), w_dq


def _check(out: torch.Tensor, ref: torch.Tensor):
    # Bounds relative to the output magnitude, as in the grouped W8A16 block-scale
    # test: the bf16 output alone rounds by up to 2^-8 of |ref|.
    assert out.dtype == torch.bfloat16
    error = (out.float() - ref).abs()
    assert error.mean() <= ref.abs().mean() * 1e-2
    assert error.max() <= ref.abs().max() * 2e-2


@pytest.mark.arch("xe20")
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("scale_kind", ["per_tensor", "block"])
def test_fp8_w8a16_gemm(m, n, k, scale_kind):
    if scale_kind == "block" and k % BLOCK != 0:
        pytest.skip("128x128 block scales need K % 128 == 0")
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="xpu") * 0.05
    quant = _quant_per_tensor if scale_kind == "per_tensor" else _quant_block
    w_fp8, scale, w_dq = quant(w)

    out = fp8_w8a16_gemm(x, w_fp8, scale)
    _check(out, x.float() @ w_dq.t())


@pytest.mark.arch("xe20")
@pytest.mark.parametrize("n,k", [(4608, 2048)])
def test_fp8_w8a16_gemm_bias_and_out(n, k):
    torch.manual_seed(0)
    m = 7
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="xpu") * 0.05
    bias = torch.randn(n, dtype=torch.float32, device="xpu")
    w_fp8, scale, w_dq = _quant_per_tensor(w)

    out = torch.empty(m, n, dtype=torch.bfloat16, device="xpu")
    ret = fp8_w8a16_gemm(x, w_fp8, scale, bias=bias, out=out)
    assert ret.data_ptr() == out.data_ptr()
    _check(out, x.float() @ w_dq.t() + bias)


@pytest.mark.arch("xe20")
def test_fp8_w8a16_gemm_empty_input():
    w_fp8, scale, _ = _quant_per_tensor(
        torch.randn(256, 256, dtype=torch.bfloat16, device="xpu")
    )
    x = torch.empty(0, 256, dtype=torch.bfloat16, device="xpu")
    assert fp8_w8a16_gemm(x, w_fp8, scale).shape == (0, 256)


@pytest.mark.arch("xe20")
def test_fp8_w8a16_gemm_rejects_unaligned_n():
    w_fp8, scale, _ = _quant_per_tensor(
        torch.randn(32, 256, dtype=torch.bfloat16, device="xpu")
    )
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="xpu")
    with pytest.raises(RuntimeError, match="N must be divisible by 64"):
        fp8_w8a16_gemm(x, w_fp8, scale)


if __name__ == "__main__":
    pytest.main([__file__])
