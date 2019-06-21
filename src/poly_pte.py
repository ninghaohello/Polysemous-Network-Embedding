import numpy as np
import torch
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError
    def forward_i(self, data):
        raise NotImplementedError
    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):
    def __init__(self, vocab1_size=20000, vocab2_size=20000, num_poly=5, embedding_size=32):
        super(Word2Vec, self).__init__()
        self.vocab1_size = vocab1_size
        self.vocab2_size = vocab2_size
        self.num_embedding1 = vocab1_size * num_poly
        self.num_embedding2 = vocab2_size * num_poly
        self.embedding_size = embedding_size

        # Initialize embedding vectors
        self.vectors_1 = nn.Embedding(self.num_embedding1, self.embedding_size)
        self.vectors_2 = nn.Embedding(self.num_embedding2, self.embedding_size)
        self.vectors_1.weight = nn.Parameter(FT(self.num_embedding1, self.embedding_size).uniform_(
            -0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.vectors_2.weight = nn.Parameter(FT(self.num_embedding2, self.embedding_size).uniform_(
            -0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.vectors_1.weight.requires_grad = True
        self.vectors_2.weight.requires_grad = True

    def forward_1(self, nodes):
        v = LT(nodes.data.numpy())
        v = v.cuda() if self.vectors_1.weight.is_cuda else v
        return self.vectors_1(v)

    def forward_2(self, nodes):
        v = LT(nodes.data.numpy())
        v = v.cuda() if self.vectors_2.weight.is_cuda else v
        return self.vectors_2(v)


class PolyPTE(nn.Module):
    def __init__(self, embedding, vocab1_size, vocab2_size, num_poly, num_negs=20, weights=None):
        super(PolyPTE, self).__init__()
        self.embedding = embedding
        self.vocab1_size = vocab1_size
        self.vocab2_size = vocab2_size
        self.num_embedding1 = vocab1_size * num_poly
        self.num_embedding2 = vocab2_size * num_poly
        self.num_negs = num_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords, type_iword=1):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]

        # Decide which type of negative nodes to sample depend on each iword
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.num_negs, replacement=True).view(
                batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.num_negs).uniform_(0, self.num_embedding2 - 1).long()

        vectors_1 = self.embedding.forward_1(iword).unsqueeze(2)
        vectors_2 = self.embedding.forward_2(owords)
        vectors_neg = self.embedding.forward_2(nwords).neg()
        if context_size == 1:
            oloss = torch.bmm(vectors_2, vectors_1).squeeze().sigmoid().log().unsqueeze(1).mean(1)
        else:
            oloss = torch.bmm(vectors_2, vectors_1).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(vectors_neg, vectors_1).squeeze().sigmoid().log().view(-1, context_size, self.num_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()