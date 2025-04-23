import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from translator.dataset import TranslationsDataset
from translator.model import TranslatorModel
from translator.utils import collate_fn
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    batches = tqdm(dataloader, desc='Training', leave=False)

    for src, tgt_in, tgt_out in batches:
        src = src.to(DEVICE)
        tgt_in = tgt_in.to(DEVICE)

        tgt_out = tgt_out.to(DEVICE)
        tgt_out = tgt_out.view(-1)

        optimizer.zero_grad()
        output = model(src, tgt_in)
        output = output.view(-1, output.size(-1))

        loss = criterion(output, tgt_out)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        batches.set_postfix({ 'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

def main():
    BATCH = 32
    EPOCHS = 10
    LEARNING_RATE=1e-4

    print(f'Using device: {DEVICE}')

    dataset = TranslationsDataset('data')
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)

    model = TranslatorModel(
        src_vocab_size=dataset.pt_vocab_size,
        tgt_vocab_size=dataset.en_vocab_size,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = tqdm(range(EPOCHS), desc='Epochs')

    for epoch in epochs:
        avg_loss = train(model, dataloader, optimizer, criterion)
        print(f"Epoch {epoch}. Loss: {avg_loss:.4f}")

        test = model.translate('Oi, como vai?', dataset, DEVICE)
        print(f"Test translation: {test}")

if __name__ == '__main__':
    main()