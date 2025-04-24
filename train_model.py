import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.optim import Adam
from translator.dataset import TranslationsDataset
from translator.model import TranslatorModel
from translator.utils import collate_fn
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
print(f'Using device: {DEVICE}')

wandb.login()

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
    defaults=dict(
        dropout=0.1,
        n_epochs=10,
        n_layers=6,
        n_heads=8,
        learning_rate=1e-4,
        batch_size=32,
    )

    wandb.init(project='translator-attn', config=defaults)
    config = wandb.config
    print(f"Config: {config}")

    dataset = TranslationsDataset('data')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    model = TranslatorModel(
        src_vocab_size=dataset.pt_vocab_size,
        tgt_vocab_size=dataset.en_vocab_size,
        dropout=config.dropout,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    ).to(DEVICE)

    wandb.watch(model, log='all', log_graph=True)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = tqdm(range(config.n_epochs), desc='Epochs')

    for epoch in epochs:
        avg_loss = train(model, dataloader, optimizer, criterion)
        print(f"Epoch {epoch}. Loss: {avg_loss:.4f}")

        test = model.translate('Oi, como vai?', dataset, DEVICE)
        print(f"Test translation: {test}")

        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
        })

        if epoch % 2 == 0:
            checkpoint = f'model_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint)
            wandb.save(checkpoint)

if __name__ == '__main__':
    main()