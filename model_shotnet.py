import torch.nn as nn
import torch
from model_resnet import Resnet3
from torch.nn import functional as F
import math
import inspect
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class Shot(nn.Module):
    def __init__(self,chan_word,width=64,pictures=8,n_embd=128):
        super().__init__()
        self.chan_word = chan_word
        self.width = width
        self.pictures = pictures
        self.n_embd = n_embd
        # define mapping on the immage according to content
        self.keys = nn.Parameter(nn.Linear(width*width*pictures,n_embd,bias=False).weight.reshape(pictures,width*width,n_embd))
        self.query = nn.Linear(chan_word,pictures*n_embd)
    def forward(self,x):
        B,T,C = x.shape
        queries = self.query(x).view(B,T,self.pictures,self.n_embd,1)
        
        prod = self.keys@queries / math.sqrt(self.n_embd)
        attention = F.softmax(prod.reshape(B,T,self.pictures,self.width*self.width),dim=-1)*self.width
        return attention

class ShotNet(nn.Module):
    def __init__(self,n_emb:int =512,vocab_size: int = 50304 ,block_size: int = 1024):
        super().__init__()
        self.block_size = 1024
        self.wpe = nn.Embedding(self.block_size,n_emb)
        self.wte = nn.Embedding(vocab_size,n_emb)
        
        # 3 traditional GPT layers to fine tune tokens

        self.block1 = self._make_gpt_layer(n_emb,vocab_size,block_size)

        self.block2 = self._make_gpt_layer(n_emb,vocab_size,block_size)
        
        self.block3 = self._make_gpt_layer(n_emb,vocab_size,block_size)
        self.n_pictures = 8
        self.pic_width = 20
        self.vocab_size= vocab_size
        self.image_mapping = Shot(n_emb,self.pic_width,self.n_pictures) ## chan_word = n_embd intrun transformer, foarte confuzing
        
        

        self.resnet = Resnet3(self.n_pictures,vocab_size)

    def _make_gpt_layer(self,n_emb,vocab_size,block_size):
        config = GPTConfig(block_size,vocab_size,None,8,n_emb,0,False)
        return Block(config)

    def forward(self,x,targets=None):
        device = x.device
        b,t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device) 

        x = self.wpe(pos)+self.wte(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        images = self.image_mapping(x)
        ## now we simply add image[i] and image[i+1] to image[i+1] to ensure connectivity btwn tokens
        images_clone = torch.empty_like(images)
        for i in range(1,t):
            images_clone[:,i,:,:]  = images[:,i,:,:] + images[:,i-1,:,:]


        logits = self.resnet(images_clone.reshape(b*t,self.n_pictures,self.pic_width,self.pic_width))
        logits = logits.reshape(b,t,self.vocab_size)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits,loss
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    model = ShotNet()
    copy = model.wpe.weight.clone().detach()
    words = torch.randint(0,900,(10,100))
    targets = torch.randint(0,900,(10,100))
    optimizer  = torch.optim.AdamW(model.parameters())

    _,loss= model(words,targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
