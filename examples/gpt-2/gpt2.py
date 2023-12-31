"""
[picoGPT](https://github.com/jaymody/picoGPT)
[gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch)
"""
import json
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import GenerationConfig

from torch import nn

from sampling import *


def layer_norm(tokens, w, b):
    # ln = nn.LayerNorm(normalized_shape=tokens.shape[1])
    # ln.weight.data = w
    # ln.bias.data = b
    # r = ln(tokens)
    res = (tokens - torch.mean(tokens, dim=1).unsqueeze(-1)) / (torch.std(tokens, dim=1).unsqueeze(-1) + 1e-8)
    res = res * w + b
    return res


class Attention(nn.Module):
    def __init__(self, pth, sub_i):
        super(Attention, self).__init__()
        self.n_head = 12

        self.attn_bias = pth[f'h.{sub_i}.attn.bias']
        self.attn_c_attn_weight = pth[f'h.{sub_i}.attn.c_attn.weight']
        self.attn_c_attn_bias = pth[f'h.{sub_i}.attn.c_attn.bias']
        self.attn_c_proj_weight = pth[f'h.{sub_i}.attn.c_proj.weight']
        self.attn_c_proj_bias = pth[f'h.{sub_i}.attn.c_proj.bias']

    def _attention(self, q, k, v):
        w = q @ k
        w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.attn_bias[0, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def _split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if is_key:
            return x.permute(1, 2, 0)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(1, 0, 2)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, x):
        x = x.permute(1, 0, 2).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def forward(self, tokens, past_cache=None):
        qkv = tokens @ self.attn_c_attn_weight + self.attn_c_attn_bias
        q, k, v = qkv.split(tokens.shape[1], dim=1)

        if past_cache is None:
            kv_cache = (k, v)
        else:
            k_cache, v_cache = past_cache[0], past_cache[1]
            k = torch.cat([k_cache, k], -2)
            v = torch.cat([v_cache, v], -2)
            kv_cache = (k, v)

        q = self._split_heads(q)
        k = self._split_heads(k, True)
        v = self._split_heads(v)
        a = self._attention(q, k, v)
        a = self._merge_heads(a)
        a = a @ self.attn_c_proj_weight + self.attn_c_proj_bias
        return a, kv_cache


class MLP(nn.Module):
    def __init__(self, pth, sub_i):
        super(MLP, self).__init__()
        self.mlp_c_fc_weight = pth[f'h.{sub_i}.mlp.c_fc.weight']
        self.mlp_c_fc_bias = pth[f'h.{sub_i}.mlp.c_fc.bias']
        self.mlp_c_proj_weight = pth[f'h.{sub_i}.mlp.c_proj.weight']
        self.mlp_c_proj_bias = pth[f'h.{sub_i}.mlp.c_proj.bias']

    def forward(self, tokens):
        tokens = tokens @ self.mlp_c_fc_weight + self.mlp_c_fc_bias
        tokens = F.gelu(tokens)
        res = tokens @ self.mlp_c_proj_weight + self.mlp_c_proj_bias
        return res


class Block(nn.Module):
    def __init__(self, pth, sub_i):
        super(Block, self).__init__()
        self.ln_1_weight = pth[f'h.{sub_i}.ln_1.weight']
        self.ln_1_bias = pth[f'h.{sub_i}.ln_1.bias']
        self.attn = Attention(pth, sub_i)
        self.ln_2_weight = pth[f'h.{sub_i}.ln_2.weight']
        self.ln_2_bias = pth[f'h.{sub_i}.ln_2.bias']
        self.mlp = MLP(pth, sub_i)

    def forward(self, x, past_cache=None):
        x_attn, kv_cache = self.attn(layer_norm(x, self.ln_1_weight, self.ln_1_bias), past_cache)
        x = x + x_attn
        x_mlp = self.mlp(layer_norm(x, self.ln_2_weight, self.ln_2_bias))
        x = x + x_mlp
        return x, kv_cache


class TinyGPT2(nn.Module):
    def __init__(self, pth):
        super(TinyGPT2, self).__init__()
        self.wte = pth['wte.weight']
        self.wpe = pth['wpe.weight']
        self.layers = nn.ModuleList()
        for i in range(12):
            self.layers.append(Block(pth, i))
        self.ln_f_weight = pth['ln_f.weight']
        self.ln_f_bias = pth['ln_f.bias']

    def forward(self, tokens_ids, pos_ids, n_past=0, past_cache=None):
        tokens_embed = self.wte[tokens_ids]
        position_embed = self.wpe[pos_ids]

        tokens = tokens_embed + position_embed

        if past_cache is None:
            past_cache = [None] * 12

        present_cache = []
        for i in range(12):
            tokens, kv_cache = self.layers[i](tokens, past_cache[i])
            present_cache.append(kv_cache)
        tokens = layer_norm(tokens, self.ln_f_weight, self.ln_f_bias)
        logits = tokens @ self.wte.permute(1, 0)
        return logits, present_cache


def openai_pipeline(text, max_token):
    model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir='../pretrained')
    cfg = GenerationConfig.from_pretrained(
        pretrained_model_name='gpt2',
        cache_dir='../pretrained',
        max_new_tokens=max_token,
        top_k=10,
        temperature=0.8
    )

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    output = generator(text, generation_config=cfg)
    print(output[0]['generated_text'])


def tinygpt2_pipeline(text, max_token):
    encoding = tokenizer(text)
    token_ids = encoding['input_ids']

    tiny_gpt = TinyGPT2(pth)

    with open(base_path + '/vocab.json', 'r') as fp:
        decoder = json.load(fp)
    dec = Decoder(decoder)
    kv_cache = None
    n_past = 0
    print(text)
    for i in range(max_token):
        token, kv_cache = sample(
            model=tiny_gpt,
            past_cache=kv_cache,
            n_past=n_past,
            token_ids=token_ids,
            temperature=0.8,
            top_k=10
        )
        # token_ids.append(token.numpy().item())
        token_ids = [token.numpy().item()]
        output = dec.decode(token_ids)
        n_past += 1
        print(output, end='')


if __name__ == '__main__':
    base_path = '../../pretrained/gpt2'

    tokenizer = AutoTokenizer.from_pretrained('/home/lzz/CLionProjects/llawa/pretrained/gpt2')
    pth = torch.load(f"{base_path}/pytorch_model.bin")

    text = "This is a story about "

    tinygpt2_pipeline(text, max_token=200)
    # openai_pipeline(text, max_token=200)
