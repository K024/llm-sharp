import torch
from torch import Tensor


def pack_u4(x: Tensor, order: list[int] = None) -> Tensor:
    assert x.dtype == torch.int32
    dim1, dim2 = x.shape
    packed = torch.zeros(dim1, (dim2 + 7) // 8, dtype=torch.int32, device=x.device)
    for n in range(8):
        i = order[n] if order is not None else n
        packed[:, :] |= (x[:, i::8] & 0xF) << (n * 4)
    return packed


def unpack_u4(x: Tensor, order: list[int] = None) -> Tensor:
    assert x.dtype == torch.int32
    dim1, dim2 = x.shape
    unpacked = torch.zeros(dim1, dim2 * 8, dtype=torch.int32, device=x.device)
    for n in range(8):
        i = order[n] if order is not None else n
        unpacked[:, i::8] = (x >> (n * 4)) & 0xF
    return unpacked


def quant_u4(x: Tensor, group_size=128, symmetric=False):
    out_dim, in_dim = x.shape
    x = x.reshape(out_dim, in_dim // group_size, group_size)

    if not symmetric:
        max, _ = torch.max(x, -1, keepdim=True)
        min, _ = torch.min(x, -1, keepdim=True)
    else:
        max, _ = torch.max(x.abs(), -1, keepdim=True)
        min = -max

    scales = torch.clamp(max - min, min=1e-10) / 15
    zeros = torch.nn.functional.relu(-min)

    x = torch.clamp(torch.round((x + zeros) / scales + 0.5), 0, 15).to(torch.int32).reshape(out_dim, in_dim)
    zeros = torch.clamp(torch.round(zeros / scales + 0.5), 0, 15).to(torch.int32).reshape(out_dim, in_dim // group_size)

    return x, zeros, scales.reshape(out_dim, in_dim // group_size)


def dequant_u4(x: Tensor, zeros: Tensor, scales: Tensor, group_size=128):
    out_dim, in_dim = x.shape
    x = x.reshape(out_dim, in_dim // group_size, group_size)
    zeros = zeros.reshape(out_dim, in_dim // group_size, 1)
    scales = scales.reshape(out_dim, in_dim // group_size, 1)
    return ((x - zeros) * scales).reshape(out_dim, in_dim)


if __name__ == "__main__":
    import math

    x = torch.randn(10, 2048)
    a = torch.randn(1024, 2048) / math.sqrt(2048)

    qa, qzeros, scales = quant_u4(a, symmetric=True)
    deq_a = dequant_u4(unpack_u4(pack_u4(qa)), qzeros, scales)

    assert (qzeros == 8).all().item()

    assert (a - deq_a).abs().max().item() < 0.05
    assert (a - deq_a).abs().mean().item() < 0.005

    order = [0, 2, 4, 6, 1, 3, 5, 7]
    assert (unpack_u4(pack_u4(qa, order), order) == qa).all().item()

    print("All tests passed!")
