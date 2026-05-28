import functools
import math

import torch
from torch import Tensor


def hyperbolic_float32(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        input_dtypes = []
        for arg in args:
            if isinstance(arg, Tensor):
                input_dtypes.append(arg.dtype)

        float32_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                float32_args.append(arg.float() if arg.dtype != torch.float32 else arg)
            else:
                float32_args.append(arg)

        float32_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, Tensor):
                float32_kwargs[key] = value.float() if value.dtype != torch.float32 else value
            else:
                float32_kwargs[key] = value

        result = func(*float32_args, **float32_kwargs)

        if isinstance(result, tuple):
            restored = []
            for res in result:
                if isinstance(res, Tensor):
                    target_dtype = input_dtypes[0] if input_dtypes else res.dtype
                    restored.append(res.to(target_dtype))
                else:
                    restored.append(res)
            return tuple(restored)
        if isinstance(result, Tensor):
            target_dtype = input_dtypes[0] if input_dtypes else result.dtype
            return result.to(target_dtype)
        return result

    return wrapper


@hyperbolic_float32
def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    x_time = lorentz_time(x, curv=curv)
    y_time = lorentz_time(y, curv=curv)
    return x @ y.transpose(0, 1) - x_time @ y_time.transpose(0, 1)


@hyperbolic_float32
def pairwise_dist(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    c_xyl = -curv * pairwise_inner(x, y, curv=curv)
    distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return distance / torch.sqrt(curv)


@hyperbolic_float32
def lorentz_time(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    return torch.sqrt(torch.clamp(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True), min=eps))


@hyperbolic_float32
def point_to_hyperplane_scores(
    x: Tensor,
    normal_space: Tensor,
    normal_time: Tensor,
    bias: Tensor,
    curv: float | Tensor = 1.0,
    eps: float = 1e-6,
):
    if x.dim() != 2 or normal_space.dim() != 2:
        raise ValueError(
            f'x and normal_space must be rank-2 tensors, got x.dim={x.dim()}, normal_space.dim={normal_space.dim()}'
        )

    if normal_space.size(1) != x.size(1):
        raise ValueError(
            f'normal_space feature dim {normal_space.size(1)} must match x feature dim {x.size(1)}'
        )

    if normal_time.dim() != 1 or normal_time.size(0) != normal_space.size(0):
        raise ValueError(
            f'normal_time shape {tuple(normal_time.shape)} must match num hyperplanes {normal_space.size(0)}'
        )

    if bias.dim() != 1 or bias.size(0) != normal_space.size(0):
        raise ValueError(f'bias shape {tuple(bias.shape)} must match num hyperplanes {normal_space.size(0)}')

    x_time = lorentz_time(x, curv=curv, eps=eps)
    minkowski_numer = x @ normal_space.transpose(0, 1) - x_time @ normal_time.unsqueeze(0) + bias.unsqueeze(0)
    hyperplane_scale = torch.sqrt(
        torch.clamp(normal_time ** 2 - torch.sum(normal_space ** 2, dim=-1), min=eps)
    )
    return minkowski_numer / hyperplane_scale.unsqueeze(0)


@hyperbolic_float32
def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    rc_xnorm = torch.sqrt(curv) * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
    return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)


@hyperbolic_float32
def smooth_clip_tangent_norm(x: Tensor, max_norm: float | Tensor = 2.0, eps: float = 1e-6):
    max_norm_tensor = torch.as_tensor(max_norm, device=x.device, dtype=x.dtype)
    if torch.all(max_norm_tensor <= 0):
        return x

    while max_norm_tensor.dim() < x.dim():
        max_norm_tensor = max_norm_tensor.unsqueeze(-1)

    max_norm_tensor = max_norm_tensor.clamp_min(eps)
    raw_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    bounded_norm = max_norm_tensor * torch.tanh(raw_norm / max_norm_tensor)
    return x * (bounded_norm / raw_norm)


@hyperbolic_float32
def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    rc_x_time = torch.sqrt(torch.clamp(1 + curv * torch.sum(x ** 2, dim=-1, keepdim=True), min=1 + eps))
    distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = torch.sqrt(curv) * torch.norm(x, dim=-1, keepdim=True)
    return distance0 * x / torch.clamp(rc_xnorm, min=eps)
