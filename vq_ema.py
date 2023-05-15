import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

def empty_init(*shape):
    #return torch.empty(shape)
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t
    
def collect_embeddings(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b t -> h b t d', d = dim) # extend(repeat) indicies with head dimensin
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch) # extend(repeat) key embeddings with batch dimension
    return embeds.gather(2, indices) # gather closest keys from codebook based on indeces 

class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        n_heads,
        heads_dim,
        codebook_size,
        decay
    ):
        super().__init__()
        self.input_dim=input_dim
        self.n_heads=n_heads
        self.decay = decay
        self.codebook_size = codebook_size
        
        key_embed = empty_init(n_heads,codebook_size,heads_dim) # init codebooks
        #print("key embed initted ",key_embed[0])
        
        self.register_buffer('key_embed', key_embed)  # register as non weight, but still part of model
        self.register_buffer('key_embed_avg', key_embed.clone()) # register as non weight, but still part of model
        
    def forward(
        self,
        x,
        key_optim
    ):
    
        x = x.float()
        shape, dtype = x.shape, x.dtype
        
        if self.n_heads>1:
            ein_rhs_eq = 'h b t d' # h-head, b-batch, t-token, d-heads dimension
            x = rearrange(x, f'b t (h d) -> {ein_rhs_eq}', h = self.n_heads) # segment input into heads
        
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d') # merge the batch and token dimensions         
        
        emb = self.key_embed
        #print("input shape ",shape, "flatten shape ", flatten.shape, "embed shape ", emb.shape)

        dist = -torch.cdist(flatten, emb, p = 2)  # calculate euclidean distance
        
        #print("dist ",dist.shape)        
        
        emb_ind = dist.argmax(dim= -1) # save indices of closest keys for each head
        emb_onehot = F.one_hot(emb_ind, self.codebook_size).type(dtype) # one hot encoding for ( head, token, codebook index 1hot )
        #print("emb ind ", emb_ind.shape)
        emb_ind = emb_ind.view(*shape[:-1])
        
        #print("emb_onehot ", emb_onehot.shape)
        
        quantized = collect_embeddings(emb_ind, emb) # collect closest key for each head
        
        if key_optim:
            emb_sum = einsum('h n d, h n c -> h c d', flatten, emb_onehot) # weigthed sum of embeddings
            self.key_embed.data.lerp_(emb_sum, self.decay)
        
        #print("ke ",self.key_embed.data[0][0])
        
        quantized = rearrange(quantized, 'h b t d -> b t (h d)' , h=self.n_heads) # concatenate the segments back together
        emb_ind = rearrange(emb_ind, 'h b n -> b n h', h=self.n_heads) # reshape indice tensor
        
        return quantized, emb_ind
        
        
        
