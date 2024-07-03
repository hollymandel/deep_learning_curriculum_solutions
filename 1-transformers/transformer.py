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
    Creates an attention mask of size NxN that is True for indices where i >= j. Because 
    the mask will be added to the key-query products prior to the application of softmax, 
    False values are represented as a large negative number and True values are represented 
    as 0.

    This mask is used in sequence processing tasks to prevent earlier tokens from attending
    to later tokens.

    Parameters:
    N (int): The size of the mask, which will be NxN.

    Returns:
    torch.Tensor: A tensor of shape (N, N) where the value is True (represented as 0) 
                  for positions where the row index is greater than or equal to the column 
                  index, and False (represented as -1e9) otherwise. 
                  
                  For example, if N = 3:
                    tensor([[-0.0000e+00, -1.0000e+09, -1.0000e+09],
                            [-0.0000e+00, -0.0000e+00, -1.0000e+09],
                            [-0.0000e+00, -0.0000e+00, -0.0000e+00]])
    """

    return (torch.range(1,N).repeat(N,1) > torch.range(1,N).repeat(N,1).transpose(0,1)
        ) * -1e9
    
def positional_encoder(M, N, R = 100):
    """
    Generates a positional encoding matrix for a sequence of length N in a M dimensional space.
    Added to input to inject information about the positions of elements in the sequence since
    the architecture itself does not encode this information.

    Parameters:
    M (int): The number of dimensions for each position encoding, equal to the model embedding
            dimension. Must be an even number.
    N (int): The length of the sequence.
    R (int, optional): A scaling factor for the positional encoding. Should be much larger 
                       than M to ensure sufficient separation between different positions.
                       Defaults to 100.

    Returns:
    torch.Tensor: A tensor of shape (N, M) containing the positional encodings. Each row
                  corresponds to the positional encoding for one position in the sequence.
    """
    r = np.power(R,2/M)
    ts = torch.arange(0,N).repeat(int(M/2),1)
    ks = torch.transpose(torch.range(0,M/2-1).repeat(N,1), 0,1)
    thetas = ts/np.power(r,ks)
    return torch.stack(
        (torch.sin(thetas),torch.cos(thetas)), dim=1).view(M,N).transpose(0,1)

class AttnHead(nn.Module):
    """
    Attention Head module for implementing scaled dot-product attention. The attention
    head computes the attention scores and applies them to the value vectors to obtain
    the output.

    Attributes:
    M (int): The representation dimension of the input and output vectors.
    N (int): The length of the input sequence.
    masked (bool): If True, applies a mask to prevent attention to future positions.
    WQ (nn.Linear): Learnable linear layer to project inputs to query vectors.
    WK (nn.Linear): Learnable linear layer to project inputs to key vectors.
    WV (nn.Linear): Learnable linear layer to project inputs to value vectors.
    mask (torch.Tensor): Mask to be applied if `masked` is True.

    Methods:
    dot_product_attn(Q, K, V):
        Computes the attention scores by taking the dot product between queries Q
        and keys K. Then uses the attention scores to linearly combine the values V.
    forward(x):
        Projects the input through linear layers to obtain Q, K, V, and then applies
        the dot-product attention.
    """
    def __init__(self, M, N, masked = False):
        """
        Initializes the AttnHead module.

        Parameters:
        M (int): The representation dimensional of the input and output vectors.
        N (int): The length of the input sequence.
        masked (bool, optional): If True, applies a mask to prevent attention to future
                                 positions. Defaults to False.
        """
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
        Computes the dot-product attention with optional masking.

        Parameters:
        Q (torch.Tensor): Query vectors of shape (b, N, M).
        K (torch.Tensor): Key vectors of shape (b, N, M).
        V (torch.Tensor): Value vectors of shape (b, N, M).

        Returns:
        torch.Tensor: Output after applying attention, of shape (b, N, M).
        """
        KQ = torch.matmul(Q, torch.transpose(K,1,2)) / np.sqrt(self.M)
        if self.masked:
            KQ = KQ + self.mask.repeat([KQ.shape[0],1,1])
        KQ = torch.nn.Softmax(2)(KQ)
        return torch.matmul(KQ,V) 
        
    def forward(self, x):
        """
        Forward pass through the attention head.

        Parameters:
        x (torch.Tensor): Input tensor of shape (b, N, M).

        Returns:
        torch.Tensor: Output tensor after applying attention, of shape (b, N, M).
        """
        Q, K, V = self.WQ(x), self.WK(x), self.WV(x)
        return self.dot_product_attn(Q,K,V)
    
class MultiHead(nn.Module):
    """
    Multi-Head Attention module for running multiple attention heads in parallel and
    linearly combining their outputs using a learned fully-connected layer.

    Attributes:
    masked (bool): If True, applies a mask to prevent attention to future positions.
    O (nn.Linear): Learned linear layer to project the concatenated outputs of the 
        attention heads.
    ahs (nn.ModuleList): List of attention heads (AttnHead instances).

    Methods:
    forward(x):
        Applies the multi-head attention to the input.
    """
    def __init__(self, M, N, d_out = None, masked = False, n_heads = 1):
        """
        Initializes the MultiHead module.

        Parameters:
        M (int): The dimensionality of the input and output vectors.
        N (int): The length of the input sequence.
        d_out (int, optional): The dimensionality of the output vectors. Defaults to M.
        masked (bool, optional): Parameter passed to AttnHead module. If True, attention heads
                                 apply a causal mask.Defaults to False.
        n_heads (int, optional): The number of attention heads. Defaults to 1.
        """
        super().__init__()
        d_out = d_out or M
        self.masked = masked
        self.O = nn.Linear(in_features = M * n_heads, out_features = d_out)
        self.ahs = nn.ModuleList(
            [ AttnHead(M, N, masked = masked) for _ in range(n_heads) ])
        
    def forward(self, x):
        """
        Applies the multi-head attention to the input.

        Parameters:
        x (torch.Tensor): Input tensor of shape (b, N, M).

        Returns:
        torch.Tensor: Output tensor after applying multi-head attention,
                      of shape (b, N, d_out).
        """
        xp = [ ah(x) for ah in self.ahs ]
        xp = torch.cat(xp, 2)
        return self.O(xp)
    
class AttnBlock(nn.Module):
    """
    Attention Block module for combining two masked attention heads and one fully connected
    layer with layer norm in between and a dropout layer at the end. 

    Attributes:
    head1, head2 (MulltiHead): multi-head attention with masking.
    feedwordward (nn.Linear): fully connected layer
    ln1, ln2, ln3 (nn.LayerNorm): Layer normalization after the first attention head.
    dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
    forward(x):
        Applies the attention block to the input.
    """

    def __init__(self, M, N, n_heads):
        """
        Initializes the AttnBlock module.

        Parameters:
        M (int): The dimensionality of the input and output vectors.
        N (int): The length of the input sequence.
        n_heads (int, optional): The number of parallel attention heads in each Multihead. 
            Defaults to 1.
        """
        super().__init__()
        self.head1 = MultiHead(M, N, n_heads = n_heads, masked = True)
        self.ln1 = nn.LayerNorm(M)
        self.head2 = MultiHead(M, N, n_heads = n_heads, masked = True)
        self.ln2 = nn.LayerNorm(M)
        self.feedforward = nn.Linear(in_features = M, out_features = M)
        self.ln3 = nn.LayerNorm(M)
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        """
        Applies the attention block to the input.

        Parameters:
        x (torch.Tensor): Input tensor of shape (b, N, M).

        Returns:
        torch.Tensor: Output tensor after applying the attention block,
                      of shape (b, N, M).
        """
        x = x + self.head1(x)
        x = self.ln1(x)
        x = x + self.head2(x)
        x = self.ln2(x)
        x = x + self.feedforward(x)
        x = self.ln3(x)
        x = self.dropout(x)
        return x
    
class Transformer(nn.Module):
    """
    Transformer model for causal sequence-to-sequence tasks.

    Attributes:
    embed_dim (int): The dimensionality of the input embeddings (None for no embedding layer).
    embedding_layer (nn.Linear): Linear layer to project input embeddings to the model dimension.
    embedding_ln (nn.LayerNorm): Layer normalization for the input embeddings.
    positional_encoder (torch.Tensor): Positional encoding matrix.
    attnBlocks (nn.ModuleList): List of attention blocks.
    linear (nn.Linear): Linear layer to project the final output to the target dimensionality.

    Methods:
    forward(x):
        Applies the Transformer model to the input.
    """
    def __init__(
        self, 
        N, 
        M, 
        n_blocks = 1,
        n_heads = 1,  
        d_target = None,
        embed_dim = None,
    ): 
        """
        Initializes the Transformer model.

        Parameters:
        N (int): The number of tokens in the input sequence.
        M (int): The representation dimension of the model.
        n_blocks (int, optional): The number of attention blocks. Defaults to 1.
        n_heads (int, optional): The number of attention heads in each block. Defaults to 1.
        d_target (int, optional): The dimensionality of the output vectors. Defaults to M.
        embed_dim (int, optional): The dimension of the input vector. If provided, an embedding 
                                   layer is added to transform from embed_dim to M. If omitted, 
                                   there is no input embedding, and the dimension of inputs must
                                   already be M.
        """
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
        """
        Applies the Transformer model to the input.

        Parameters:
        x (torch.Tensor): Input tensor of shape (b, N, embed_dim) if embed_dim is provided,
                          otherwise of shape (b, N, M).

        Returns:
        torch.Tensor: Output tensor of shape (b, N, d_target).
        """
        if self.embed_dim:
            x = self.embedding_layer(x)
            x = self.embedding_ln(x)
        x = x + self.positional_encoder.repeat([x.shape[0],1,1])
        for ab in self.attnBlocks:
            x = x + ab(x)
        x = self.linear(x)
        return x