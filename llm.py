import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.is_available(), ", version:", torch.backends.cudnn.version(), ", bf32:", torch.backends.cudnn.allow_tf32)
device = "cuda" if torch.cuda.is_available() else "cpu"

LayerNorm = torch.nn.LayerNorm

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, nHeads, encodingSize, hasBias, dropout=0.0):
        super(CausalSelfAttention, self).__init__()
        self.dropout = dropout
        self.c_attn = torch.nn.Linear(encodingSize, encodingSize * 3, bias=hasBias, dtype=torch.float32)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.nHeads = nHeads

    def forward(self, xs):
        B,T,C = xs.shape
        q,k,v = self.c_attn(xs).view(B, T, self.nHeads * 3, C // self.nHeads).transpose(1, 2).split(self.nHeads, dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(xs.shape)
        return self.resid_dropout(y)

class MLP(torch.nn.Module):
    def __init__(self, encodingSize, hasBias, dropout=0.0):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(OrderedDict([
            ("c_fc",torch.nn.Linear(encodingSize, encodingSize * 4, bias=hasBias, dtype=torch.float32)),
            ("gelu",torch.nn.GELU()),
            ("c_proj",torch.nn.Linear(encodingSize * 4, encodingSize, bias=hasBias, dtype=torch.float32)),
            ("dropout",torch.nn.Dropout(dropout))
        ]))
        self.encodingSize = encodingSize

    def forward(self, xs):
        return self.net(xs)

class Block(torch.nn.Module):
    def __init__(self, nHeads, encodingSize, hasBias, dropout=0.0):
        super(Block, self).__init__()
        self.ln1 = torch.nn.LayerNorm(encodingSize, dtype=torch.float32)
        self.attn = CausalSelfAttention(nHeads, encodingSize, hasBias, dropout)
        self.ln2 = torch.nn.LayerNorm(encodingSize, dtype=torch.float32)
        self.mlp = MLP(encodingSize, hasBias, dropout)
        if device == "cuda":
            self.to(device)

    def forward(self, xs):
        xs = self.ln1(xs)
        attn = self.attn(xs)
        xs = xs + attn
        ln2 = self.ln2(xs)
        xs = self.mlp(xs)
        return xs + ln2
       
class EmbeddingModel(torch.nn.Module):
    def __init__(self, vocabSize, blockSize, encodingSize):
        super(EmbeddingModel, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocabSize, encodingSize, dtype=torch.float32)
        self.pos_emb = torch.nn.Embedding(blockSize, encodingSize, dtype=torch.float32)
        self.positions = torch.arange(0, blockSize, dtype=torch.int64, device=device, requires_grad=False)
        if device == "cuda":
            self.to(device)

    def forward(self, input):
        tok_emb = self.tok_emb(input)
        pos_emb = self.pos_emb(self.positions)
        return tok_emb + pos_emb


class LanguageModel(torch.nn.Module):
    def __init__(self, nLayers, nHeads, nEmbed, vocabSize, blockSize, hasBias=False, dropout=0.0):
        super().__init__()
        self.blockSize = blockSize
        self.vocabSize = vocabSize
        layers = OrderedDict()
        layers["embed"] = EmbeddingModel(vocabSize, blockSize, nEmbed)
        for i in range(1,nLayers+1):
            layers["block" + str(i)] = Block(nHeads, nEmbed, hasBias, dropout)
        layers["de_embed"] = torch.nn.Linear(nEmbed, vocabSize, dtype=torch.float32)
        self.layers = torch.nn.Sequential(layers)
        if device == "cuda":
            self.to(device)

    def forward(self, input):
        #print("input shape:", input.shape)
        return self.layers.forward(input)

def generateIxs(model, ixs, nTokens):
    model.eval()
    result = [ixs]
    for i in range(1, nTokens+1):
        #yHat = model.forward(ixs)
        yHat = model(ixs)
        yHat = yHat.select(1,-1)
        probs = torch.nn.functional.softmax(yHat, 1)
        yIdx = torch.multinomial(probs, num_samples=1)
        ixs = torch.cat([ixs[:, 1:], yIdx], dim=1)
        result.append(yIdx)
    return torch.cat(result, dim=1)

def get_loss(model, ys, yHat):
    batchSize = ys.shape[0]
    logits = yHat.reshape(-1, model.vocabSize)
    target = ys.reshape([batchSize * model.blockSize])
    loss = torch.nn.functional.cross_entropy(logits, target)
    return loss


#model = Block(4, cfg.n_embd, False)
#t = torch.rand([1,8,cfg.n_embd], device=device, dtype=torch.float32)
#model.forward(t)
