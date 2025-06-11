import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from CheckersModel import GameTransformer
from CheckersDataset import CheckersDataset, CheckersStreamDataset, checkersEvaluator  # Import checkersEvaluator
from pytorch_lightning.callbacks import ModelCheckpoint
# from Checkers.Database.SQL_checkers import get_positions # Not directly used here anymore, CheckersStreamDataset handles it
import torch

import os
import re

# Konfiguracja CUDA dla Tensor Cores
torch.set_float32_matmul_precision("high")

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
    # Make sure you have enough data in your DB for these chunk sizes
    train_ds = CheckersStreamDataset(chunk_size=200,  start_offset=0)
    val_ds = CheckersStreamDataset(chunk_size=200, start_offset=30000)

    # It's important that num_workers > 0 for IterableDataset and persistent_workers=True
    # If you encounter issues with multiprocessing (especially on Windows), try num_workers=0 or 1 first.
    train_loader = DataLoader(train_ds, batch_size=64, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4, persistent_workers=True)

    checkpoint_path = get_best_checkpoint_nested("checkpoints/")

    # Updated max_len and num_moves based on checkers logic
    # num_embeddings in GameTransformer (for Embedding layer) needs to be max_piece_value + 1
    # Our pieces are 0,1,2,3,4. So num_embeddings=5
    # Our board_tensor is 101 long (100 fields + 1 color)
    # Our output num_moves is 50*50 = 2500

    if checkpoint_path:
        print(f"Wczytywanie modelu z checkpointu: {checkpoint_path}")
        model = GameTransformer.load_from_checkpoint(checkpoint_path, evaluator=checkersEvaluator)  # Pass evaluator
    else:
        print("Brak checkpointów – tworzenie nowego modelu.")
        model = GameTransformer(
            d_model=512,
            max_len=101,  # 100 board positions + 1 for active player
            num_moves=2500,  # Max possible move indices (50 * 50)
            num_heads=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            lr=3e-4,
            evaluator=checkersEvaluator  # Pass the evaluator function
        )

    checkpoint_cb = ModelCheckpoint(
        monitor='val/loss',  # metryka do monitorowania
        dirpath='checkpoints/',  # folder, gdzie będą zapisywane pliki
        filename='checkers-transformer-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3,  # trzy najlepsze wg val/loss
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
        devices="auto",  # jak dużo gpu/cpu zostanie uzytych
        precision="32",  # FP16 mixed precision
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=20,
        limit_train_batches=50,  # DO TESTÓW
        limit_val_batches=10,  # DO TESTÓW
        # fast_dev_run=1
        # max_steps=10 # This will override max_epochs for testing, remove for full training
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':  # Changed from '__main_checkers__' to '__main__'
    torch.multiprocessing.freeze_support()
    main()