import pytorch_lightning as pl
from torch.utils.data import DataLoader
from CheckersTraining.CheckersDataset import CheckersDataset


class CheckersDataModule(pl.LightningDataModule):
    def __init__(self, chunk_size: int = 200, records_per_epoch: int = 1000, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        self.records_per_epoch = records_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        self.train_ds = CheckersDataset(chunk_size=self.chunk_size, records_per_epoch=self.records_per_epoch)
        self.val_ds = CheckersDataset(chunk_size=self.chunk_size, records_per_epoch=self.records_per_epoch)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False)

    def on_train_epoch_start(self):

        ep = self.trainer.current_epoch
        self.train_ds.current_epoch = ep

    def state_dict(self):
        return {
            "train_ds_state": self.train_ds.state_dict(),
            "val_ds_state": self.val_ds.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.train_ds.load_state_dict(state_dict["train_ds_state"])
        self.val_ds.load_state_dict(state_dict["val_ds_state"])

