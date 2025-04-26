import torch
import torch.nn as nn
import wandb
import tempfile
import os
from tqdm import tqdm
from torch.optim import Adam
from translator.dataset import TranslationsDataset
from translator.ray_dataset import get_ray_dataset, get_tokenizers
from translator.model import TranslatorModel
from translator.utils import collate_fn, collate_ray_fn
from torch.utils.data import DataLoader
import ray.train.torch

wandb.login()

def initialize(config, device, use_ray=False):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(local_dir, 'data')

    pt_tokenizer, en_tokenizer = get_tokenizers(dataset_dir)

    model = TranslatorModel(
        src_vocab_size=pt_tokenizer.tokenizer.get_vocab_size(),
        tgt_vocab_size=en_tokenizer.tokenizer.get_vocab_size(),
        dropout=config.dropout,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )

    if use_ray:
        # Prepare model
        ray.train.torch.prepare_model(model)

        # Prepare dataloader
        dataset = get_ray_dataset(dataset_dir)
        dataloader = dataset.iter_torch_batches(
            batch_size=config.batch_size,
            collate_fn=collate_ray_fn,
        )

        # Prepare metrics reporter
        def ray_reporter(dir, metrics):
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(dir),
            )

        return model, dataloader, ray_reporter, pt_tokenizer, en_tokenizer
    else:
        model.to(device)

        dataset = TranslationsDataset(dataset_dir)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

        return model, dataloader, None, pt_tokenizer, en_tokenizer


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_count = 0

    batches = tqdm(dataloader, desc='Training', leave=False)

    for src, tgt_in, tgt_out in batches:
        if device:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

        tgt_out = tgt_out.view(-1)

        optimizer.zero_grad()
        output = model(src, tgt_in)
        output = output.view(-1, output.size(-1))

        loss = criterion(output, tgt_out)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        total_count += 1
        batches.set_postfix({ 'loss': f'{loss.item():.4f}'})

    return total_loss / total_count

def train(use_ray=False):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

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

    model, dataloader, ray_reporter, pt_tokenizer, en_tokenizer = initialize(
        config,
        DEVICE,
        use_ray=use_ray,
    )

    wandb.watch(model, log='all', log_graph=True)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = tqdm(range(config.n_epochs), desc='Epochs')

    for epoch in epochs:
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device=DEVICE if not use_ray else None)
        print(f"Epoch {epoch}. Loss: {avg_loss:.4f}")

        test = model.translate('Oi, como vai?', pt_tokenizer, en_tokenizer, DEVICE)
        print(f"Test translation: {test}")

        metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'test_prediction': test
        }

        wandb.log(metrics)

        if use_ray and ray.train.get_context().get_world_rank() == 0:
            print(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f'{tmpdir}/model_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            alias = [f"epoch-{epoch}", f"loss-{avg_loss:.4f}"]

            model_artifact = wandb.Artifact(
                name="attn-translator",
                type="model",
                metadata=metrics
            )

            model_artifact.add_file(checkpoint_path, name='model.pt')

            wandb.log_artifact(model_artifact, aliases=alias)

            if ray_reporter is not None:
                ray_reporter(tmpdir, metrics)

if __name__ == '__main__':
    train()
