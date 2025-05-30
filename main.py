import torch
from Model.BetterModel import ChessTransformer
from Chess.Database.SQL_chess import get_positions


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

matplotlib.use('TkAgg')

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "Model" / "lightning_logs"


version_dirs = sorted(glob.glob(str(LOG_DIR / "version_*")))
if not version_dirs:
    raise FileNotFoundError("Nie znaleziono folderów lightning_logs/version_*")
latest_dir = version_dirs[-1]

csv_path = os.path.join(latest_dir, 'metrics.csv')
df = pd.read_csv(csv_path)

print(df)

print(df["step"])
print(df["train/loss"])

# train = df[df["train/loss"].notna()]
# val_step = df[df["val/loss_step"].notna()]
# val_epoch = df[df["val/loss_epoch"].notna()]
#
#plt.figure()

train_acc_percent = df["train/acc"]*100
val_acc_percent = df["val/acc"]*100

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))


ax1.plot(df['step'], df['train/loss'], label='Train Loss')
ax1.plot(df['step'], df['val/loss'], label='Val Loss (step)', linestyle='dotted')
ax1.set_xlabel('Krok treningowy (step)')
ax1.set_ylabel('Wartość loss')
ax1.legend()

ax2.plot(df['step'], train_acc_percent, label='Train Accuracy')
ax2.plot(df['step'], val_acc_percent, label='Val Accuracy', linestyle='dotted')
ax2.set_ylabel('Wartość accuracy [%]')
ax2.legend()

plt.title('Loss podczas treningu i walidacji')
plt.tight_layout()
plt.show()

