from transformers import BertModel
from torch import nn
from einops import rearrange 

from dkv_bn import DiscreteKeyValueBottleneck

class BERTwithBottleNeck(nn.Module):
    def __init__(self,pool_before,pooling_type,n_labels):
        super(BERTwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels
        
        self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
            encoder = self.encoder,   # pass the frozen encoder into the bottleneck
            decoder = 'values_softmax', # type of decoder: 1 layer mlp or values_softmax(non parametric)
            dim = 768,                # input dimension
            num_key_segments = 64, # number of key segments
            codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
            dim_key = 12,        # dimension of the key segments
            dim_value = self.n_labels,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
            decay = 0.8,              # the exponential moving average decay, lower means the keys will change faster
            pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
            pooling_type = self.pooling, # type of pooling : cls or mean
            n_labels = self.n_labels
        )
        
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask,token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,key_optim=key_optim)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        
        #         
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
        if self.pool_before:
           outputs = rearrange(outputs, 'b 1 d -> b d')
               
        if self.enc_with_bottleneck.decoder=='mlp':
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)        
        if self.enc_with_bottleneck.decoder=='values_softmax':
            logits = outputs.mean(dim=1)       

        return logits

class BERTbase(nn.Module):
    def __init__(self,n_labels):
        super(BERTbase, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        
        
    def forward(self, ids, mask, token_type_ids):
        outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        dropout_output = self.l2(outputs[1])
        logits = self.l3(dropout_output)
        
        return logits