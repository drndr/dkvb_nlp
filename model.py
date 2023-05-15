from transformers import BertModel
from torch import nn
from einops import rearrange 

from dkv_bn import DiscreteKeyValueBottleneck

class BERTwithBottleNeck(nn.Module):
    def __init__(self,pool_before,n_labels):
        super(BERTwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        self.n_labels = n_labels
        
        self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
            encoder = self.encoder,   # pass the frozen encoder into the bottleneck
            dim = 768,                # input dimension
            num_memory_codebooks = 64, # number of heads
            num_memories = 4096,   # number of memories
            dim_memory = 12,        # dimension of the output memories
            decay = 0.1,              # the exponential moving average decay, lower means the keys will change faster
            pool_before = self.pool_before
        )
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask,token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,key_optim=key_optim)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        if not self.pool_before:
            outputs = outputs[:,0] # Pool CLS token here
        dropout_output = self.l2(outputs)
        logits = self.l3(dropout_output)
        if self.pool_before:
           logits = rearrange(logits, 'b 1 h -> b h')
        return logits     