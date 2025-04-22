import torch
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from translator.tokenizer import BertTokenizer

class TranslationsDataset(Dataset):
    def __init__(self, path, max_tokens=145):
        """
        Assumes that path contains the {en|pt}-tokens.json tokenizers data (see prepare_data.ipynb) 
        and the generated dataset.tsv training set
        """

        super().__init__()
        self.max_tokens = max_tokens
        self.data = pd.read_csv(join(path, 'dataset.tsv'), sep='\t')

        print(f"Loaded dataset with size: {len(self.data)}")

        self.pt_tokenizer = BertTokenizer(join(path, 'pt-tokens.json'))
        self.en_tokenizer = BertTokenizer(join(path, 'en-tokens.json'))

    def tokenize_pt(self, sentence):
        return self.pt_tokenizer.encode(sentence)
    
    def tokenize_en(self, sentence):
        return self.en_tokenizer.encode(sentence)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pt_text = self.data['pt'][index]
        en_text = self.data['pt'][index]

        pt_tokens = self.tokenize_pt(pt_text).ids[:self.max_tokens]
        en_tokens = self.tokenize_en(en_text).ids[:self.max_tokens]

        return torch.tensor(pt_tokens), torch.tensor(en_tokens)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', default='data')
    args = parser.parse_args()

    # Test basic loading and encoding
    dataset = TranslationsDataset(args.p)

    print(dataset.tokenize_en('Hi, how are you?').tokens)
    print(dataset.tokenize_pt('Oi, como vai?').tokens)

    print(dataset.__getitem__(0))
