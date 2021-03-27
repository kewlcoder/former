import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, wide=False):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        # x.shape = (batch_size, max_length, emb_dim)
        # For classification example, it is (4, 512, 128)
        # print("before pooling x.shape = ", x.shape)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension(average/pooling over max_length i.e. length of sequence)
        
        # x.shape = (batch_size, emb_dim)
        # output -> (4, 128)
        # print("after pooling x.shape = ", x.shape)

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)