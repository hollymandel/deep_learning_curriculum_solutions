import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
N is sequence length
M is the dimensionality of the embedding space. Must be even for positional encoder. 
b is the batch size
"""

def create_mask(N):
    """
    returns a NxN mask that is true for i <= j
    """
    return (torch.range(1,N).repeat(N,1) > torch.range(1,N).repeat(N,1).transpose(0,1)
        ) * -1e9
    
def positional_encoder(M, N, R = 100):
    """
    R is a parameter of the positional encoder. 
    R should be much larger than M. Wikipedia calls R -> N.
    """
    r = np.power(R,2/M)
    ts = torch.arange(0,N).repeat(int(M/2),1)
    ks = torch.transpose(torch.range(0,M/2-1).repeat(N,1), 0,1)
    thetas = ts/np.power(r,ks)
    return torch.stack(
        (torch.sin(thetas),torch.cos(thetas)), dim=1).view(M,N).transpose(0,1)

class AttnHead(nn.Module):
    def __init__(self, M, N, masked = False):
        super().__init__()
        self.WQ = nn.Linear(in_features = M, out_features = M, bias = False)
        self.WK = nn.Linear(in_features = M, out_features = M, bias = False)
        self.WV = nn.Linear(in_features = M, out_features = M, bias = False)
        self.masked = masked
        if masked:
            self.mask = create_mask(N)
        self.M = M
        self.N = N
 
    def dot_product_attn(self, Q, K, V):
        """
        Q, K should be b x N x M
        V should be b x N x M
        """
        KQ = torch.matmul(Q, torch.transpose(K,1,2)) / np.sqrt(self.M)
        if self.masked:
            KQ = KQ + self.mask.repeat([KQ.shape[0],1,1])
        KQ = torch.nn.Softmax(2)(KQ)
        return torch.matmul(KQ,V) 
        
    def forward(self, x):
        Q, K, V = self.WQ(x), self.WK(x), self.WV(x)
        return self.dot_product_attn(Q,K,V)
    
class MultiHead(nn.Module):
    def __init__(self, M, N, d_out = None, masked = False, n_heads = 1):
        super().__init__()
        d_out = d_out or M
        self.masked = masked
        self.O = nn.Linear(in_features = M * n_heads, out_features = d_out)
        self.ahs = nn.ModuleList(
            [ AttnHead(M, N, masked = masked) for _ in range(n_heads) ])
        
    def forward(self, x):
        xp = [ ah(x) for ah in self.ahs ]
        xp = torch.cat(xp, 2)
        return self.O(xp)
    
class AttnBlock(nn.Module):
    def __init__(self, M, N, n_masked_heads = 1, n_unmasked_heads = 1):
        super().__init__()
        self.head1 = MultiHead(M, N, n_heads = n_masked_heads, masked = True)
        self.ln1 = nn.LayerNorm(M)
        self.head2 = MultiHead(M, N, n_heads = n_unmasked_heads, masked = True)
        self.ln2 = nn.LayerNorm(M)
        self.feedforward = nn.Linear(in_features = M, out_features = M)
        self.ln3 = nn.LayerNorm(M)
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        x = x + self.head1(x)
        x = self.ln1(x)
        x = x + self.head2(x)
        x = self.ln2(x)
        x = x + self.feedforward(x)
        x = self.ln3(x)
        x = self.dropout(x)
        return x
    
class Transformer(nn.Module):
    def __init__(
        self, 
        N, 
        M, 
        n_blocks = 1,
        n_heads = 1,  
        d_target = None,
        embed_dim = None,
    ): 
        super().__init__()
        d_target = d_target or M
        self.embed_dim = embed_dim
        if self.embed_dim:
            self.embedding_layer = nn.Linear(embed_dim, M, bias=False)
            self.embedding_ln = nn.LayerNorm(M)
        self.positional_encoder = positional_encoder(M, N)
        self.attnBlocks = nn.ModuleList([ AttnBlock(M, N, n_masked_heads = n_heads, n_unmasked_heads = n_heads) for _ in range(n_blocks) ])
        self.linear = nn.Linear(M, d_target)

    def forward(self, x):
        if self.embed_dim:
            x = self.embedding_layer(x)
            x = self.embedding_ln(x)
        x = x + self.positional_encoder.repeat([x.shape[0],1,1])
        for ab in self.attnBlocks:
            x = x + ab(x)
        x = self.linear(x)
        return x