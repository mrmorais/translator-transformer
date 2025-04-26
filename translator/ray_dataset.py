from translator.tokenizer import BertTokenizer
from pyarrow import csv
import ray
from typing import Dict
import torch
import os
from translator.utils import collate_ray_fn
from functools import partial

def get_tokenizers(dir = 'data/'):
    pt_tokenizer = BertTokenizer(os.path.join(dir, 'pt-tokens.json'))
    en_tokenizer = BertTokenizer(os.path.join(dir, 'en-tokens.json'))

    return pt_tokenizer, en_tokenizer

def tokenize_rows(pt_tokenizer, en_tokenizer, data: Dict[str, str]) -> Dict[str, list]:
        pt_text = data['pt']
        en_text = data['en']

        pt_tokens = pt_tokenizer.encode(pt_text).ids
        en_tokens = en_tokenizer.encode(en_text).ids

        return { 'pt': pt_tokens, 'en': en_tokens }

def get_ray_dataset(dir = 'data/'):
    pt_tokenizer, en_tokenizer = get_tokenizers(dir)

    parse_options = csv.ParseOptions(delimiter='\t')
    data = ray.data.read_csv(os.path.join(dir, 'dataset.tsv'), parse_options=parse_options).limit(1000)

    data = data.map(partial(tokenize_rows, pt_tokenizer, en_tokenizer))

    return data

if __name__ == '__main__':
    ray.init()

    data = get_ray_dataset()
    dataloader = data.iter_torch_batches(
        batch_size=2,
        collate_fn=collate_ray_fn,
    )

    itere = iter(dataloader)
    print(itere.__next__())