import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


logger = logging.getLogger(__name__)

class SupportsMetadata :
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.metadata = kwargs
    @classmethod
    def from_metadata(cls, metadata) :
        return cls(**metadata)

class HoleFillingBiLinear (nn.Module, SupportsMetadata) :
    r"""Applies linear, bilinear and trilinear local constraints to predict
    center token from left and right tokens: :math:`Pr(y | x_l, x_r) \propto
    \exp(x_l B y + y B x_r + a y)`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): the size of each embedding vector.
        share_weights (bool): If set to False, left, right and center words will
            maintain different embeddings.
            Default: ``True``

    Shape:
        Input: :math:`(batch_size, 2)``, the index of left and right token.
        Output: :math:`(batch_size, \text{num_embeddings})``.
    """

    def __init__(self,
        num_embeddings : int, embedding_dim : int,
        share_weights : bool = True,
    ) :
        super().__init__()
        # self.num_embeddings = num_embeddings
        # self.embedding_dim = embedding_dim
        # self.share_weights = share_weights
        SupportsMetadata.__init__(self,
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            share_weights = share_weights,
        )

        self.emb = nn.Embedding(
            num_embeddings = self.num_embeddings,
            embedding_dim = self.embedding_dim,
        )
        ## self.emb.weight : torch.Tensor(self.num_embeddings, self.embedding_dim)
        if self.share_weights :
            self.emb_ctx_left = self.emb
            self.emb_ctx_right = self.emb
        else :
            self.emb_ctx_left = nn.Embedding(
                num_embeddings = self.num_embeddings,
                embedding_dim = self.embedding_dim,
            )
            self.emb_ctx_right = nn.Embedding(
                num_embeddings = self.num_embeddings,
                embedding_dim = self.embedding_dim,
            )
        # self.linear = nn.Linear(
        #     in_features = self.embedding_dim,
        #     out_features = 1,
        #     bias = False,
        # )
        # self.bilinear = nn.Bilinear(
        #     in1_features = self.embedding_dim,
        #     in2_features = self.embedding_dim,
        #     out_features = 1,
        #     bias = False,
        # )
        # self.trilinear = nn.Bilinear(
        #     in1_features = self.embedding_dim,
        #     in2_features = self.embedding_dim,
        #     out_features = self.embedding_dim,
        #     bias = False,
        # )
        self.weight_linear = Parameter(torch.Tensor(
            self.embedding_dim,
        ))
        ##  (out_features x in_features)
        self.weight_bilinear = Parameter(torch.Tensor(
            self.embedding_dim, # left
            self.embedding_dim, # right
        ))
        # ##  (out_features x in1_features x in2_features)
        # self.weight_trilinear = Parameter(torch.Tensor(
        #     self.embedding_dim, # middle
        #     self.embedding_dim, # left
        #     self.embedding_dim, # right
        # ))
        self.final_activation = torch.nn.LogSoftmax(dim = -1)

    def reset_parameters(self):
        self.emb.reset_parameters()
        if not self.share_weights :
            self.emb_ctx_left.reset_parameters()
            self.emb_ctx_right.reset_parameters()
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.weight_linear.data.uniform_(-stdv, stdv)
        self.weight_bilinear.data.uniform_(-stdv, stdv)
        # self.weight_trilinear.data.uniform_(-stdv, stdv)

    def forward(self, ctx : torch.LongTensor) :
        ## ctx : torch.LongTensor(batch_size, 2)
        logger.debug('ctx.size(): {}'.format(ctx.size()))
        ctx_left, ctx_right = ctx[:, 0], ctx[:, 1]
        ## ctx_left, ctx_right : torch.LongTensor(batch_size, 1)
        emb_ctx_left, emb_ctx_right = self.emb_ctx_left(ctx_left), self.emb_ctx_right(ctx_right)
        ## emb_ctx_left, emb_ctx_right : torch.Tensor(batch_size, ctx_embedding_dim)
        logger.debug('emb_ctx_left.size(): {}'.format(emb_ctx_left.size()))
        # F.linear(x, A) computes $y = x A^T + b$
        # Input: (N,∗,in_features) where ∗ means any number of additional dimensions
        # Output: (N,∗,out_features)
        bilinear_left = F.linear(emb_ctx_left, self.weight_bilinear.t())
        bilinear_right = F.linear(emb_ctx_left, self.weight_bilinear)
        # ## bilinear_left, bilinear_right : torch.Tensor(batch_size, embedding_dim)
        # # F.bilinear computes $y = x_1 A x_2 + b$
        # # Input: (N,∗,in1_features), (N,∗,in2_features) where ∗ means any number of additional dimensions
        # # Output: (N,∗,out_features)
        # trilinear = F.bilinear(emb_ctx_left, emb_ctx_right, self.weight_trilinear)
        # ## trilinear : torch.Tensor(batch_size, embedding_dim)
        # logger.debug('trilinear.size(): {}'.format(trilinear.size()))
        weight_ctx = self.weight_linear + bilinear_left + bilinear_right# + trilinear
        ## weight_ctx : torch.Tensor(batch_size, embedding_dim)
        score = F.linear(weight_ctx, self.emb.weight)
        ## score : torch.Tensor(batch_size, self.num_embeddings)
        logger.debug('score.size(): {}'.format(score.size()))
        output = self.final_activation(score)
        ## output : torch.Tensor(batch_size, self.num_embeddings)
        logger.debug('output.size(): {}'.format(output.size()))
        return output
