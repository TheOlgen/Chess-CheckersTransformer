import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from ChessDataset import ChessDataset, ChessStreamDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from Chess.Database.SQL_chess import get_positions
import torch
import re
import os
import sys

from Model.ChessDataset import chessEvaluator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Model.BetterModel import GameTransformer



def get_best_checkpoint_nested(root_dir="checkpoints/"):
    pattern = re.compile(r"loss=([0-9.]+)\.ckpt")

    best_loss = float('inf')
    best_ckpt_path = None

    if not os.path.exists(root_dir):
        return None

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".ckpt"):
                    match = pattern.search(file)
                    if match:
                        loss = float(match.group(1))
                        if loss < best_loss:
                            best_loss = loss
                            best_ckpt_path = os.path.join(subdir_path, file)

    return best_ckpt_path


def main():
    train_ds = ChessStreamDataset(chunk_size=200)
    val_ds = ChessStreamDataset(chunk_size=200)

    train_loader = DataLoader(train_ds, batch_size=64, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds,   batch_size=64, num_workers=4, persistent_workers=True)

    checkpoint_path = get_best_checkpoint_nested("checkpoints/")

    if checkpoint_path:
        print(f"Wczytywanie modelu z checkpointu: {checkpoint_path}")
        model = GameTransformer.load_from_checkpoint(checkpoint_path)
    else:
        print("Brak checkpointów – tworzenie nowego modelu.")
        model = GameTransformer(
            d_model=512,
            max_len=72,
            num_moves=4096,
            num_embeddings=13,
            num_heads=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            lr=3e-4,
            evaluator=chessEvaluator,
        )

    checkpoint_cb = ModelCheckpoint(
        monitor='val/loss',            # metryka do monitorowania
        dirpath='checkpoints/',        # folder, gdzie będą zapisywane pliki
        filename='chess-transformer-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3,                  # trzy najlepsze wg val/loss
        mode='min'

    )

    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",     #jak dużo gpu/cpu zostanie uzytych
        precision="16",              # FP16 mixed precision
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=20
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
