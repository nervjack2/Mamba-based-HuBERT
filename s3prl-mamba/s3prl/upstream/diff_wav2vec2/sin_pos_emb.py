# This file is copied from 
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
import math
from typing import Any, Optional
import torch
import torch.onnx.operators
from fairseq import utils
from torch import nn, Tensor

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """
    def __init__(self, embedding_dim, padding_idx=None, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            ),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore some deprecated keys that were used in older versions
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            # If the input is longer than the number of pre-computed embeddings,
            # compute the extra embeddings on the fly.
            # Only store the expanded embeddings if auto_expand=True.
            # In multithreading environments, mutating the weights of a module
            # may cause trouble. Set auto_expand=False if this happens.
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        )

# def make_positions(pad_mask, padding_idx):
#     mask = torch.where(~pad_mask==True, 1, 0)
#     return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

# class SinusoidalPositionalEmbedding(nn.Module):
#     """
#     This module produces sinusoidal positional embeddings of any length.
#     Padding symbols are ignored.
#     """
#     def __init__(self, embedding_dim, padding_idx=0, init_size=1024):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx if padding_idx is not None else 0
#         self.weights = SinusoidalPositionalEmbedding.get_embedding(
#             init_size, embedding_dim, padding_idx
#         )
        
#         self.register_buffer("_float_tensor", torch.FloatTensor(1))
#         self.max_positions = int(1e5)

#     @staticmethod
#     def get_embedding(
#         num_embeddings, embedding_dim, padding_idx=0
#     ):
#         """Build sinusoidal embeddings.
#         This matches the implementation in tensor2tensor, but differs slightly
#         from the description in Section 3.5 of "Attention Is All You Need".
#         """
#         half_dim = embedding_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
#         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
#             1
#         ) * emb.unsqueeze(0)
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
#             num_embeddings, -1
#         )
#         if embedding_dim % 2 == 1:
#             # zero pad
#             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
#         if padding_idx is not None:
#             emb[padding_idx, :] = 0
#         return emb

#     def forward(
#         self,
#         pad_mask,
#     ):
#         """Pad mask is expected to be of size [bsz x seqlen]."""
#         bsz, seq_len = pad_mask.shape[0], pad_mask.shape[1]
        
#         max_pos = self.padding_idx + 1 + seq_len
#         if self.weights is None or max_pos > self.weights.size(0):
#             # recompute/expand embeddings if needed
#             self.weights = SinusoidalPositionalEmbedding.get_embedding(
#                 max_pos, self.embedding_dim, self.padding_idx
#             )
#         self.weights = self.weights.to(self._float_tensor)

#         positions = make_positions(
#             pad_mask, self.padding_idx
#         )
        
#         return (
#             self.weights.index_select(0, positions.view(-1))
#             .view(bsz, seq_len, -1)
#             .detach()
#         )