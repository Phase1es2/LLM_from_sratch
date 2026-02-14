import torch
import torch.nn as nn
from einops import einsum

from cs336_basics.utils.xavier_init import xavier_init

#(a, b) * (b, c) == (a, c)

class Linear(nn.Module):
    """
    Implement a Linear class that inherits from torch.nn.Module, and
    performs a linear transformation. the implementation should follow
    the interface of PyTorch's nn.Linear class, module, except for not having
    a bias argument or parameter.
    """
    def __init__(
            self,
            in_features: int,                       # final dimension of the input
            out_features: int,                      # final dimension of the output
            device: torch.device | None = None,     # Device to store the parameters on
            dtype: torch.dtype | None = None        # Data type of the parameters
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = xavier_init(self.in_features, self.out_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the linear transformation to the input
            # y = x (W.T)   [batch_size, in_features] dot [out_features, in_features]
        """
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")