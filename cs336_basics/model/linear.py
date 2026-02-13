import math
import torch
import torch.nn as nn

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the linear transformation to the input
        """
