import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from BetterModel import ChessTransformer
from ChessDataset import ChessDataset
from pytorch_lightning.callbacks import ModelCheckpoint


#TODO: Automatyczne pobieranie z bazy danych
fens = [...]
best_moves = [...]

dataset = ChessDataset(fens, best_moves)

val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

model = ChessTransformer(
    d_model=512,
    max_len=64,
    num_moves=4096,
    num_heads=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    lr=3e-4
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
    gpus=1,                    # lub liczba GPU, które masz
    precision=16,              # jeżeli chcesz FP16
    callbacks=[checkpoint_cb, early_stop_cb],
    log_every_n_steps=20
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)





