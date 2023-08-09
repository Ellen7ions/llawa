import json
from typing import Dict, List, Any
from enum import Enum
import torch
import argparse
import pickle


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
    word: str


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

    vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab = [LlawaVocab(length=len(v), word=v) for v, _id in vocab]
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
        vocab=vocab,
        tensors=tensors
    )


if __name__ == '__main__':
    base_path = '../pretrained/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=base_path, type=str, help="load model path")
    parser.add_argument("--output", default='llawa_gpt2.bin', type=str, help="output model path")
    args = parser.parse_args()

    pth = torch.load(f'{base_path}/pytorch_model.bin')

    with open(f"{base_path}/config.json", 'r') as f:
        config = json.load(f)

    with open(f"{base_path}/vocab.json", 'r') as f:
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
        for k in dir(llawa_file.hparams):
            if not k.startswith('__'):
                f.write(getattr(llawa_file.hparams, k).to_bytes(length=4, byteorder='little', signed=False))

        print('writing n_vocab...')
        f.write(llawa_file.n_vocab.to_bytes(length=4, byteorder='little', signed=False))

        print('writing vocabs...')
        for vb in llawa_file.vocab:
            f.write(vb.length.to_bytes(length=4, byteorder='little', signed=False))
            f.write(vb.word.encode())

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
