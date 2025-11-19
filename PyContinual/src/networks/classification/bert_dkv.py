import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from transformers import BertModel, BertConfig

####################################
# Helper functions
####################################
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper functions for VQ    
def empty_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t
    
def collect_embeddings(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b t -> h b t d', d = dim) # extend(repeat) indicies with head dimensin
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch) # extend(repeat) key embeddings with batch dimension
    return embeds.gather(2, indices) # gather closest keys from codebook based on indeces 
    
#####################################
# DKV Bottleneck
#####################################

#BERT with DKV        
class Net(nn.Module):

    def __init__(self,taskcla,args):
        super(Net, self).__init__()
        
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.args = args
        self.encoder = BertModel.from_pretrained(args.bert_model,config=config)
        
        self.pool_before = False
        self.pooling = "mean"
        self.n_labels = 2
        
        self.decoder = "softmax"  # change here for decoder
        self.dim_key = 64
        self.num_key_segments = int(768/self.dim_key)
        
        self.taskcla = taskcla
        self.args = args
        
        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,                   # pass the frozen encoder into the bottleneck
                decoder = 'values_softmax',               # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,                     # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,                   # dimension of the key segments
                dim_value = self.n_labels,                # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,                              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before,           # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling,              # type of pooling : cls or mean
                n_labels = self.n_labels,                 # number of labels
                taskcla = self.taskcla,                   # task ids contained here
                args = args                               # args from config
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,                   # pass the frozen encoder into the bottleneck
                decoder = 'mlp',                          # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,                     # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,                   # dimension of the key segments
                dim_value = self.dim_key,                 # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,                              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before,           # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling,              # type of pooling : cls or mean
                n_labels = self.n_labels,                 # number of labels
                taskcla = self.taskcla,                   # task ids contained here
                args = args                               # args from config
            )
        
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        
        # Set architecture for single and multi-head configurations - for parametric setup
        if 'dil' in args.scenario:
            self.l3=torch.nn.Linear(args.bert_hidden_size,768)
        elif 'til' in args.scenario:
            self.l3=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.l3.append(torch.nn.Linear(args.bert_hidden_size,self.n_labels))
                
                
        #print("model architecture ",self)

    def forward(self,input_ids, segment_ids, input_mask, key_optim, t_id):
        output_dict = {}
        outputs = self.enc_with_bottleneck(x=input_ids, token_type_ids=segment_ids, mask=input_mask, key_optim=key_optim, t_id=t_id)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        
        # Do pooling after bottleneck    
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
        
        # Parametric decoding
        if self.enc_with_bottleneck.decoder=='mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            outputs = self.dropout(outputs)
            
            # Parametric decoding single-head vs multi-head pass
            if 'dil' in self.args.scenario:
                y = self.l3(outputs)
            elif 'til' in self.args.scenario:
                y = self.l3[t_id](outputs)
        # Non-parametric decoding          
        if self.enc_with_bottleneck.decoder=='values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            y = outputs.mean(dim=1)

                
        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(outputs, dim=1)

        return output_dict

class ValuesLayer(nn.Module):
    def __init__(self, num_key_segments, codebook_size, dim_value):
        super(ValuesLayer, self).__init__()
        self.values_layer = nn.Parameter(torch.randn(num_key_segments, codebook_size, dim_value))

    def forward(self, x):
        # no forward happening in values layer, segments are returned based on discrete key mappings
        pass

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_key_segments = 64,  # number of key segments
        codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
        dim_key = 12,           # dimension of the key segments
        dim_value = 12,         # dimension of the value segments
        decay = 1,              # decay for the EMA updates on keys
        encoder = None,         # encoder backbone
        decoder = None,         # type of decoder (parametric or non-parametric)
        pool_before = False,    # pool before or after bottleneck
        pooling_type = "mean",   # type of pooling
        n_labels = 1,           # number of labels (for last layer)
        taskcla = None,         # task ids contained here
        args = None,            # args from config
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pool_before = pool_before
        self.pooling_type = pooling_type
        self.n_labels = n_labels
        self.taskcla = taskcla
        self.args = args
        
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
        

        # For non-parametric multi-head configuration
        if self.decoder == 'values_softmax' and 'til' in args.scenario:
            self.values=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.values.append(ValuesLayer(num_key_segments, codebook_size, dim_value))
        
        # For parametric multi-head and single-head configurations
        else:
            self.values = ValuesLayer(num_key_segments, codebook_size, dim_value)
                
        print("DKV initialized ", self)

    def forward(
        self,
        x,
        mask,
        token_type_ids,
        key_optim,
        t_id,
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

        if 'til' in self.args.scenario and self.decoder == 'values_softmax':
            # Same as with einops: values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
            values = self.values[t_id].values_layer.unsqueeze(0).expand(memory_indices.shape[0], -1, -1, -1)
        elif 'dil' in self.args.scenario or self.decoder=='mlp':    
            values = self.values.values_layer.unsqueeze(0).expand(memory_indices.shape[0], -1, -1, -1)
            
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
        
# Vector Quantization Module
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
        #print("key embed initted ",key_embed[0].shape)
        
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
            #x = rearrange(x, 'b t d -> b h t d', h = self.n_heads) Segment on tokens???
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
            emb_sum = einsum('h n d, h n c -> h c d', flatten, emb_onehot) # elementwise multiplication and summation over axis n
            self.key_embed.data.lerp_(emb_sum, self.decay)
        
        #print("ke ",self.key_embed.data[0][0])
        
        quantized = rearrange(quantized, 'h b t d -> b t (h d)' , h=self.n_heads) # concatenate the segments back together
        emb_ind = rearrange(emb_ind, 'h b n -> b n h', h=self.n_heads) # reshape indice tensor
        
        return quantized, emb_ind
        