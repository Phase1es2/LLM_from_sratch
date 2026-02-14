import torch
import torch.nn as nn
from torch.nn import factory_kwargs


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,                          # Hidden dimension of the model
            eps: float = 1e-5,                     # Epsilon value for numerical stability
            device: torch.device  | None = None,   # Device to store the parameters on
            dtype: torch.dtype | None = None,      # Data type of the parameters
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        # for  RMSNOrm weight is 1
        # y = x_i / RMS(x) * g_i
        # 如果用 empy 未初始化記憶
        # 可能是 [ 3.42e+12, -1.1e-30, 9.8e-10, ... ]
        # hidden state 放大縮小幾十倍
        # 訓練剃度不穩定
        self.weight = nn.Parameter(
            torch.ones(d_model, **factory_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process an input tensor of shape (batch_size, seq_len, d_model)
        # and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        assert x.dtype == torch.float32
        rsm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # 如果模擬高興太 float16/ float16 那麼weight可能是 float16 的
        # 如果 float32 * float16 PyTorch 會做 type promotion
        # 可能變低精度，或者數值不穩定 float16 會精度不足，
        y = (x / rsm) * self.weight.to(torch.float32)

        return y.to(in_dtype)

