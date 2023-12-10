import json
from typing import Dict, List, Any
from enum import Enum
import torch
import argparse
import pickle


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


class Base:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FType(Enum):
    LLAWA_I8 = 0
    LLAWA_I16 = 1
    LLAWA_I32 = 2
    LLAWA_F16 = 3
    LLAWA_F32 = 4
    LLAWA_COUNT = 5


class HyperParams(Base):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    ftype: int


class LlawaTensor(Base):
    n_dims: int
    length: int
    ftype: int
    ne: List[int]
    name: str
    data: torch.Tensor


class LlawaVocab(Base):
    length: int
    word: List[Any]


class LlawaFile(Base):
    magic: str
    hparams: HyperParams
    n_vocab: int
    vocab: List[LlawaVocab]
    tensors: List[LlawaTensor]


def load_llawa_file(
        config: Dict[str, Any],
        vocab: Dict[str, int],
        model_pth: Dict[str, torch.Tensor]
) -> LlawaFile:
    hparams = HyperParams(
        n_vocab=config['vocab_size'],
        n_ctx=config['n_ctx'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        ftype=FType.LLAWA_F32.value,
    )

    # vocab = sorted(vocab.items(), key=lambda x: x[1])

    byte_decoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_decoder.items()}

    vocab_res = []
    for k in vocab:
        text = [byte_decoder[c] for c in k]
        vocab_res.append(LlawaVocab(length=len(bytearray(text)), word=text))

    tensors = [LlawaTensor(
        n_dims=len(v.shape),
        length=len(k),
        ftype=FType.LLAWA_F32.value,
        ne=v.shape,
        name=k,
        data=v
    ) for k, v in model_pth.items()]
    return LlawaFile(
        magic='awall'[::-1],
        hparams=hparams,
        n_vocab=hparams.n_vocab,
        vocab=vocab_res,
        tensors=tensors
    )


if __name__ == '__main__':
    base_path = '../../pretrained/gpt2'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=base_path, type=str, help="load model path")
    parser.add_argument("--output", default='llawa_gpt2.bin', type=str, help="output model path")
    args = parser.parse_args()

    pth = torch.load(f'{base_path}/pytorch_model.bin')

    with open(f"{base_path}/config.json", 'r') as f:
        config = json.load(f)

    with open(f"{base_path}/vocab.json", 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    llawa_file = load_llawa_file(
        config=config,
        vocab=vocab,
        model_pth=pth
    )

    with open(args.output, 'wb') as f:
        print('writing magic...')
        f.write(llawa_file.magic.encode())

        print('writing hparams...')
        f.write(llawa_file.hparams.n_vocab.to_bytes(length=4, byteorder='little', signed=False))
        f.write(llawa_file.hparams.n_ctx.to_bytes(length=4, byteorder='little', signed=False))
        f.write(llawa_file.hparams.n_embd.to_bytes(length=4, byteorder='little', signed=False))
        f.write(llawa_file.hparams.n_head.to_bytes(length=4, byteorder='little', signed=False))
        f.write(llawa_file.hparams.n_layer.to_bytes(length=4, byteorder='little', signed=False))
        f.write(llawa_file.hparams.ftype.to_bytes(length=4, byteorder='little', signed=False))

        print('writing n_vocab...')
        f.write(llawa_file.n_vocab.to_bytes(length=4, byteorder='little', signed=False))

        print('writing vocabs...')
        for i, vb in enumerate(llawa_file.vocab):
            f.write(vb.length.to_bytes(length=4, byteorder='little', signed=False))
            f.write(bytearray(vb.word))

        print('writing tensors...')
        for ts in llawa_file.tensors:
            f.write(ts.n_dims.to_bytes(length=4, byteorder='little', signed=False))
            f.write(ts.length.to_bytes(length=4, byteorder='little', signed=False))
            f.write(ts.ftype.to_bytes(length=4, byteorder='little', signed=False))
            for i in range(ts.n_dims):
                f.write(ts.ne[i].to_bytes(length=4, byteorder='little', signed=False))
            f.write(ts.name.encode())
            print(f'writing {ts.name} {ts.data.dtype}')
            ts.data.data.numpy().tofile(f)
