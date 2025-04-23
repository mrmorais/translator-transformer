import torch
from torch import Tensor
import torch.nn as nn
import math
from .utils import generate_mask, collate_fn

from .dataset import TranslationsDataset
from torch.utils.data import DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TranslatorModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()

        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src: Tensor, tgt: Tensor):
        src_padding_mask = (src == 0) # 0 is UNK
        tgt_padding_mask = (tgt == 0)

        # The shape at this point is [batch x sequence x d_model]
        src_embedded = self.pos_encoder(self.embedding_src(src)) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(self.embedding_tgt(tgt)) * math.sqrt(self.d_model)

        tgt_mask = generate_mask(tgt.size(1)).to(src.device)

        # Transpose to [sequence x batch x features]
        src_embedded = src_embedded.transpose(0, 1)
        tgt_embedded = tgt_embedded.transpose(0, 1)

        output = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        # To [batch x sequence (tgt) x features]
        output = output.transpose(0, 1)
        output = self.fc_out(output)

        return output
    
    def translate(self, src: str, dataset: TranslationsDataset, device, max_tokens=50) -> str:
        self.eval()

        print(dataset.tokenize_pt(src).ids)
        src_tensor = torch.tensor(dataset.tokenize_pt(src).ids).unsqueeze(0).to(device)
        
        # Init target with 0 = [UNK]
        tgt = torch.ones(1, 1).fill_(0).type(torch.long).to(device)

        for i in range(max_tokens):
            output = self(src_tensor, tgt)
            pred = output.argmax(2)[:, -1].item()
            tgt = torch.cat([tgt, torch.ones(1, 1).type_as(tgt).fill_(pred)], dim=1)

            if pred == 3: # [PAD]
                break
        print(tgt.squeeze(0).tolist())
        return dataset.decode_en(tgt.squeeze(0).tolist())

if __name__ == '__main__':

    dataset = TranslationsDataset('data')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    model = TranslatorModel(dataset.pt_vocab_size, dataset.en_vocab_size)
    input, out, o = dataloader._get_iterator()._next_data()
    print(input.shape, out.shape)
    model(input, out)

    print(model.translate('Oi, como vai?', dataset, 'cpu'))