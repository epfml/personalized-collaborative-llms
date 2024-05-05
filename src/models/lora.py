"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

And LoRA's implementation is inspired by:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import math
from argparse import Namespace
from typing import Union, Dict, List, Any

import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import T
from transformers import GPT2LMHeadModel


def lora_model(model: nn.Module, lora_freeze_all_non_lora: bool, lora_allow_embedding: bool) -> None:
    """ Freeze the correct parameters of the model """
    if lora_freeze_all_non_lora:
        for name, param in model.named_parameters():
            if 'lora' in name or (lora_allow_embedding and ('wte' in name or 'wpe' in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for _, module in model.named_modules():
            if isinstance(module, LoRALinear):
                for name, param in module.named_parameters():
                    if name == 'weight' or name == 'bias':
                        param.requires_grad = False
                    else:
                        param.requires_grad = True


class LoRALinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int,
                 lora_rank: int, lora_alpha: float, lora_dropout: float,
                 bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)

        self.lora_merged = False
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_scaling = lora_alpha / self.lora_rank
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.lora_A = nn.Parameter(
                torch.empty((in_features, lora_rank), device=self.weight.device))
            self.lora_B = nn.Parameter(
                torch.empty((lora_rank, out_features), device=self.weight.device))
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, 'lora_rank'):
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: Tensor) -> Tensor:
        x = super().forward(input)
        if not self.lora_merged and self.lora_rank > 0:
            x += self.lora_dropout(input) @ self.lora_A @ \
                 self.lora_B * self.lora_scaling
        return x

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        if mode:
            if self.lora_merged and self.lora_rank > 0:
                self.weight.data -= (self.lora_A @ self.lora_B).T * self.lora_scaling
                self.lora_merged = False
        else:
            if not self.lora_merged and self.lora_rank > 0:
                self.weight.data += (self.lora_A @ self.lora_B).T * self.lora_scaling
                self.lora_merged = True
        return self

    def eval(self: T) -> T:
        self.train(mode=False)
        return self


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        if config.lora_causal_self_attention:
            # key, query, value projections for all heads, but in a batch
            self.c_attn = LoRALinear(config.n_embd, 3 * config.n_embd,
                                     bias=config.bias,
                                     lora_rank=config.lora_rank,
                                     lora_alpha=config.lora_alpha,
                                     lora_dropout=config.lora_dropout)
            # output projection
            self.c_proj = LoRALinear(config.n_embd, config.n_embd,
                                     bias=config.bias,
                                     lora_rank=config.lora_rank,
                                     lora_alpha=config.lora_alpha,
                                     lora_dropout=config.lora_dropout)
        else:
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
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                 .view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        if config.lora_mlp:
            self.c_fc = LoRALinear(config.n_embd, 4 * config.n_embd, bias=config.bias,
                                   lora_rank=config.lora_rank,
                                   lora_alpha=config.lora_alpha,
                                   lora_dropout=config.lora_dropout)
            self.c_proj = LoRALinear(4 * config.n_embd, config.n_embd, bias=config.bias,
                                     lora_rank=config.lora_rank,
                                     lora_alpha=config.lora_alpha,
                                     lora_dropout=config.lora_dropout)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLoRA(nn.Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding('gpt2')

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.sequence_length, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters, excluding lora
        print("number of parameters at inference: %.2fM" % (self.get_num_params() / 1e6,))

        # turn on lora weights and off linear weights
        lora_model(self, lora_freeze_all_non_lora=config.lora_freeze_all_non_lora,
                   lora_allow_embedding=config.lora_allow_embedding)

        # report number of parameters, with lora
        print("number of trainable parameters: %.2fM" % (
            self.get_num_params(only_trainable=True) / 1e6,))

    def get_num_params(self, only_trainable: bool = False) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        if only_trainable:
            n_params = sum(p.numel() if p.requires_grad else 0 for n, p in self.named_parameters())
        else:
            n_params = sum(p.numel() if not 'lora' in n else 0 for n, p in self.named_parameters())
        return n_params

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding]) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor = None, get_logits: bool = False) -> Dict[str, Tensor]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss}

    def crop_sequence_length(self, sequence_length: int) -> None:
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :sequence_length, :sequence_length]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Namespace = None) -> "GPTLoRA":
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        for key in sd_hf.copy().keys():
            if any(key.endswith(k) for k in ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']):
                sd_hf[key] = sd_hf[key].T

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['sequence_length'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        config_args['dropout'] = 0.2

        # LoRA config
        config_args['lora_rank'] = override_args.lora_rank
        config_args['lora_alpha'] = override_args.lora_alpha
        config_args['lora_dropout'] = override_args.lora_dropout
        config_args['lora_mlp'] = override_args.lora_mlp
        config_args['lora_causal_self_attention'] = override_args.lora_causal_self_attention
        config_args['lora_freeze_all_non_lora'] = override_args.lora_freeze_all_non_lora
        config_args['lora_allow_embedding'] = override_args.lora_allow_embedding

        args = Namespace(**config_args)
        model = GPTLoRA(args)

        for name, param in model.named_parameters():
            if 'lora' in name:
                sd_hf[name] = param

        model.load_state_dict(sd_hf)
        return model

    def get_parameter_group_specs(self) -> List[Dict[str, Any]]:
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or 'lora_A' in pn or 'lora_B' in pn) and isinstance(m,
                                                                                                whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif (pn.endswith('weight') or 'lora_A' in pn or 'lora_B' in pn) and isinstance(m,
                                                                                                blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str: str, max_new_tokens: int, temperature: float = 1.0,
                             top_k: int = None) -> str:
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1, -1).to(
            self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
