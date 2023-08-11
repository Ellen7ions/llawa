import torch
from torch.nn import functional as F


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[-1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample(model, token_ids, n_past=0, past_cache=None, temperature=0.9, top_k=50):
    logits, kv_cache = model(
        token_ids,
        pos_ids=[n_past + k for k in range(len(token_ids))],
        past_cache=past_cache,
        n_past=n_past
    )
    logits = logits[-1, :] / temperature
    logits = top_k_logits(logits, k=top_k)
    log_probs = F.softmax(logits, dim=-1)
    if sample:
        prev = torch.multinomial(log_probs, num_samples=1)
    else:
        _, prev = torch.topk(log_probs, k=1, dim=-1)
    return prev, kv_cache


class Decoder:
    def __init__(self, decoder):
        self.decoder = {v: k for k, v in decoder.items()}
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8')
        return text
