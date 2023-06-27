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
        num_key_segments = 64, # number of key segments
        codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
        dim_key = 12,        # dimension of the key segments
        dim_value = 12, 
        decay = 1,
        encoder = None,
        decoder = None,
        pool_before = False,
        pooling_type = "cls",
        n_labels = 1,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pool_before = pool_before
        self.pooling_type = pooling_type
        self.n_labels = n_labels
        
        assert (dim % num_key_segments) == 0, 'embedding dimension must be divisible by number of codes'
        assert decoder =='mlp' or dim_value==self.n_labels, 'if decoder is values_softmax dim_values must equal to number of labels'
        assert decoder =='values_softmax' or (num_key_segments*dim_value)==768, 'if decoder is mlp num_key_segments*dim_value must equal to encoder output dim'
        
        self.vq = VectorQuantize(
            input_dim = dim,
            n_heads = num_key_segments,
            heads_dim = dim_key,
            codebook_size = codebook_size,
            decay = decay
        )

        self.values = nn.Parameter(torch.randn(num_key_segments, codebook_size, dim_value))

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
                    if self.pooling_type =="cls":
                        x = x[1]
                    if self.pooling_type =="mean":
                        x = x[0].mean(dim=1)                    
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
        
        memories = rearrange(memories, 'b h n d -> b n h d')
        #print("memories ",memories.shape)
        
        if self.decoder =='mlp':
            memories = rearrange(memories, 'b n h d -> b n (h d)')
        #print("memories ", memories.shape)
        
        return memories#flattened_memories