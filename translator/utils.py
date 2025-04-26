import torch
from torch.nn.utils.rnn import pad_sequence

# Handles entries with multiple sizes in the batch
def collate_fn(batch):
    pt_seqs = [item[0] for item in batch]
    en_seqs = [item[1] for item in batch]

    # Adds padding 0 at end based on the length of greater sample. 0 maps to UNK
    pt_padded = pad_sequence(pt_seqs, batch_first=True, padding_value=0)
    en_padded = pad_sequence(en_seqs, batch_first=True, padding_value=0)

    # TODO: Experiment without it
    # Add special tokens to target sequence (shift for next token prediction)
    # Add <UNK> at start and <PAD> at end
    tgt_input = torch.full((len(batch), en_padded.size(1) + 1), 1, dtype=torch.long)  # 1 is <UNK>
    tgt_input[:, 1:] = en_padded
    
    tgt_output = torch.full((len(batch), en_padded.size(1) + 1), 0, dtype=torch.long)  # 0 is <PAD>
    tgt_output[:, :-1] = en_padded

    return pt_padded, tgt_input, tgt_output

def collate_ray_fn(batch):
    pt_seqs = [torch.tensor(item) for item in batch['pt']]
    en_seqs = [torch.tensor(item) for item in batch['en']]

    # Adds padding 0 at end based on the length of greater sample. 0 maps to UNK
    pt_padded = pad_sequence(pt_seqs, batch_first=True, padding_value=0)
    en_padded = pad_sequence(en_seqs, batch_first=True, padding_value=0)

    # TODO: Experiment without it
    # Add special tokens to target sequence (shift for next token prediction)
    # Add <UNK> at start and <PAD> at end
    tgt_input = torch.full((len(en_seqs), en_padded.size(1) + 1), 1, dtype=torch.long)  # 1 is <UNK>
    tgt_input[:, 1:] = en_padded
    
    tgt_output = torch.full((len(en_seqs), en_padded.size(1) + 1), 0, dtype=torch.long)  # 0 is <PAD>
    tgt_output[:, :-1] = en_padded

    return pt_padded, tgt_input, tgt_output

# See mask.ipynb
def generate_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
