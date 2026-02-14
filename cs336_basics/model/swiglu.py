import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import nn, sigmoid, Tensor

from cs336_basics.model.linear import Linear

MUTIPLE = 64

def round_to_multiple_of_64(raw_dff: int) -> int:
    return int((raw_dff + MUTIPLE - 1) // MUTIPLE) * MUTIPLE


def set_d_ff(d_model: int) -> int:
    raw_dff = int(math.ceil(8.0 * d_model) / 3.0)
    return round_to_multiple_of_64(raw_dff)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            w1_weight: Float[Tensor, " d_ff d_model"],
            w2_weight: Float[Tensor, " d_model d_ff"],
            w3_weight: Float[Tensor, " d_ff d_model"],
            in_features: Float[Tensor, " ... d_model"],
            device: torch.device | None = None,  # Device to store the parameters on
            dtype: torch.dtype | None = None,  # Data type of the parameters
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else set_d_ff(d_model)
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)
    # ceil_to_multiple(x, k) = lower [(x + k - 1 ) /k] * k

    @staticmethod
    def silu(x: Tensor) -> Tensor:
        return x * sigmoid(x)

    #SwiGlu FFN = W_2 (silu(W_1 x) * (W_3 x) )
    def forward(
            self,
            in_features: torch.Tensor,
    ) -> torch.Tensor:
        # Linear
        a: Float[Tensor, "... d_ff"] = self.w1(in_features)
        # a = einsum(in_features, self.w1.weight, "... d_model, d_ff d_model -> ... d_ff")
        b: Float[Tensor, "... d_ff"] = self.w3(in_features)
        # b = einsum(in_features, self.w3.weight, "... d_model, d_ff d_model -> ... d_ff")
        gated: Float[Tensor, "... d_ff"] = self.silu(a) * b
        res: Float[Tensor, "... d_model"] = self.w2(gated)
        # res = einsum(gated, self.w2.weight, "... d_ff, d_model d_ff -> ... d_model")
        return res
