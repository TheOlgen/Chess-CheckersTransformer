import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from BetterModel import ChessTransformer
from ChessDataset import ChessDataset, ChessStreamDataset, chessEvaluator
from pytorch_lightning.callbacks import ModelCheckpoint
from Chess.Database.SQL_chess import get_positions
import torch


def main():
    #dataset = ChessDataset(fens, best_moves)
    # val_size = int(0.1 * len(dataset))
    # train_size = len(dataset) - val_size
    # train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_ds = ChessStreamDataset(chunk_size=200)
    val_ds = ChessStreamDataset(chunk_size=200)

    train_loader = DataLoader(train_ds, batch_size=64, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds,   batch_size=64, num_workers=4, persistent_workers=True)

    model = ChessTransformer(
        d_model=512,
        max_len=72,
        num_moves=4096,
        num_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        lr=3e-4,
        evaluator=chessEvaluator
    )



    checkpoint_cb = ModelCheckpoint(
        monitor='val/loss',            # metryka do monitorowania
        dirpath='checkpoints/',        # folder, gdzie będą zapisywane pliki
        filename='Chess-transformer-{epoch:02d}-{val/loss:.4f}',
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
        accelerator="auto",      #"gpu",
        devices="auto",     #jak dużo gpu/cpu zostanie uzytych
        precision="32",              # FP16 mixed precision
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=20,
        limit_train_batches=10, #DO TESTÓW
        limit_val_batches=10, #DO TESTÓW
        #fast_dev_run=1
        max_steps=20

    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.multiprocessing.freeze_support()
    main()


