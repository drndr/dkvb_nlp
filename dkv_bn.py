import torch

from torch import nn, einsum
from einops import rearrange, repeat

from vq_ema import VectorQuantize

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_memories,
        num_memory_codebooks,
        encoder = None,
        dim_memory = None,
        pool_before = False,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.pool_before = pool_before
        assert (dim % num_memory_codebooks) == 0, 'embedding dimension must be divisible by number of codes'
        self.vq = VectorQuantize(
            input_dim = dim,
            n_heads = num_memory_codebooks,
            heads_dim = dim_memory,
            codebook_size = num_memories,
            decay = 0.8
        )

        dim_memory = default(dim_memory, dim // num_memory_codebooks)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

    def forward(
        self,
        x,
        mask,
        token_type_ids,
        key_optim,
        **kwargs
    ):

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, mask, token_type_ids,**kwargs)
                if self.pool_before:
                    x = x[1]
                    x = rearrange(x, 'b h -> b 1 h')
                    #print(" x: ", x.shape, "\n values ", x) 
                else:
                    x = x[0]
                    
        vq_out = self.vq(x, key_optim)
        
        if key_optim: # if we are optimizing keys with ema, break forward here
            return None
            
        quantized, memory_indices = vq_out
        
        #print("quantized shape ",quantized.shape, " /n memory_indices shape :", memory_indices.shape)
        #print(" values shape ", self.values.shape)
        
        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        #print("values after reshape ", values.shape)
        
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])
        #print("memory ind ",memory_indices.shape)

        memories = values.gather(2, memory_indices)
        #print("memories ",memories.shape)

        flattened_memories = rearrange(memories, 'b h n d -> b n (h d)')
        #print("flattened memories ", flattened_memories.shape)

        return flattened_memories