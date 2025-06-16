import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from Model.ChessDataset import ChessDataset
from Model.Model import ChessTransformer


train_ds = ChessDataset(train_fens, train_best_moves)
val_ds = ChessDataset(val_fens, val_best_moves)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4)


model = ChessTransformer(
    d_model=256,
    max_len=64,       # 8×8 szachownica
    num_moves=64*64,  # 4096 możliwych ruchów
)

trainer = pl.Trainer(
    max_epochs=20,        # l. epok
    gpus=1,               # 1 - GPU, 0 - CPU
    precision=16,         # mixed precision
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            save_top_k=1,
            mode='min',
            filename='best-{epoch:02d}-{val/loss:.4f}'
        ),
        pl.callbacks.EarlyStopping(
            monitor='val/loss',
            patience=5,
            mode='min'
        )
    ],
    log_every_n_steps=50
)

# 4. Odpal trening
trainer.fit(model, train_loader, val_loader)

# (opcjonalnie) 5. Walidacja / Testowanie / Predykcja
# trainer.validate(model, val_loader)
# preds = trainer.predict(model, predict_loader)
