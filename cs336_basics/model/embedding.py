import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    The first layer of the Transformer is an embedding layer that maps integer token IDS
    into a vector space of dimension d_model. WE will implement a custom Embedding class
    that inherits from torch.nn.Module. The forward method should select the embedding
    vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model)
    using a torch.LongTensor token IDs with shape (batch_size, seq_len).
    """
    def __init__(
            self,
            num_embeddings: int, # Size of the vocabulary
            embedding_dim: int, # Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None,     # Device to store the parameters on
            dtype: torch.dtype | None = None        # Data type of the parameters
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # loopup the weight for the embedding layers
        # Loopup the embedding vectors for the given token IDs.
        return self.weight[token_ids]

