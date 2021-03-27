from former import util
from util import mask_

import torch
from torch import nn
import torch.nn.functional as F

import random, math
"""
@kewlcoder - NOTE - 
http://peterbloem.nl/blog/transformers

Strongly recommended to go through that blog post before diving into the code.

In the class SelfAttentionWide implementation, the heads are not created by splitting the embedding dimensions as is usually done.
Rather, all embedding matrices are copied/replicated to create 'h' heads.
This code is an implementation of the following blog post - 

Narrow and wide self-attention There are two ways to apply multi-head self-attention. The standard option is to cut the embedding
vector into chunks: if the embedding vector has 256 dimensions, and we have 8 attention heads, we cut it into 8 chunks of 32 
dimensions. For each chunk, we generate keys, values and queries of 32 dimensions each. This means that the matrices 
𝐖rq, 𝐖rk,𝐖rv are all 32×32.
"""

class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        # @kewlcoder - emb is the dimensionality of embeddings to be created.
        self.emb = emb
        self.heads = heads
        self.mask = mask

        # kewlcoder - "keys" weights to be trained. 
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        # kewlcoder - combine the heads to create 1 combined matrix 
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        """ @kewlcoder -
        Here,
        b: denotes the batch size
        t: denotes the max sequence length(max number of words/tokens in the sentence/input)
        e: the embedding dimensionality
        """
        b, t, e = x.size()
        
        # h: number of heads
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        """ @kewlcoder -
        x(input) when fed to a linear layer (b,t,e) * (e, e*h) => (b,t,e*h)
        """
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        """
        Since the head and batch dimension are not next to each other, we need to transpose before we reshape.
        (This is costly, but it seems to be unavoidable.)
        """
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - WHY??? Instead of dividing the dot products by sqrt(e), where e is the embedding dim, we scale the keys and values.
        #   This should be more memory efficient
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        """ @kewlcoder - we divide by sqrt(e) because in a 2D vector space if a vector has c value in each dimension, the 
        aggregate vector becomes sqrt(2) * c. Thus, for n-dim vector space it would have an impact of sqrt(n). Thus, as 
        dim(e) increases, the product would become bigger and bigger and to supress that, we divide by sqrt(e).
        For every item in the batch and for every head individually, compute dot = (Q*K/sqrt(emb))
        """
        
        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        """ @kewlcoder - Here, each element (i,j) of the sub-matrix (t,t) represents the attention weight to be given to word j for 
        calculating the weighted attention generated vector for word i in the sequence(or vice-versa).
        """
        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # @kewlcoder - perform out = dot * V
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        """
        can also use - 
        https://pytorch.org/docs/stable/generated/torch.einsum.html
        https://github.com/pbloem/former/issues/4
        """
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        """ @kewlcoder - (b,t,h*e)(h*e, e) => (b,t,e) -> We finally get attention weighted 
        embedding/hidden layer that can be used for various downstream tasks.
        """
        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        # @kewlcoder - add residual skip connection. Then do layer normalization.
        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
