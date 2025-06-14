import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


# You need to import checkersEvaluator and related functions from CheckersDataset.py
# Make sure CheckersDataset.py is accessible in your module path
# from CheckersDataset import checkersEvaluator, index_to_move, move_to_index # Might need full path

# It's better to pass the evaluator directly if it has external dependencies,
# but if it's part of the same package, a relative import might work.
# For simplicity, assume CheckersDataset is importable here.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 101):  # Updated max_len
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, d_model, 2, dtype=torch.float) * (torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            n_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor = None,
            key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:

        # Self-attention + residual + norm
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward + residual + norm
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class GameTransformer(pl.LightningModule):
    def __init__(
            self,
            d_model: int = 512,
            max_len: int = 101,  # rozmiar wejscia (100 board + 1 color)
            num_moves: int = 2500,  # ile mozliwych ruchow (50 * 50 = 2500)
            num_embeddings: int = 14,  # ile roznych wartosci tokenow
            num_heads: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            lr: float = 3e-4,
            evaluator=None,
            converter=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.evaluator = evaluator
        self.converter = converter

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_moves)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_moves)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len) -> (batch_size, 101) where 101 is 100 board + 1 color
        batch_size, seq_len = x.size()

        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)  # plus positional encoding

        device = x.device

        # attn_mask: (seq_len, seq_len), -inf na gornej czesci
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )

        key_padding_mask = None

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        #TODO: Attention pooling
        pooled = x.mean(dim=1)  # (batch_size, d_model)
        logits = self.fc(pooled)  # (batch_size, num_moves)
        return logits

    def training_step(self, batch, batch_idx):
        boards, moves = batch
        logits = self(boards)
        loss = self.loss_fn(logits, moves)
        preds = torch.argmax(logits, dim=-1)
        acc = self.train_accuracy(preds, moves)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        boards, moves = batch
        logits = self(boards)
        loss = self.loss_fn(logits, moves)
        preds = torch.argmax(logits, dim=-1)
        acc = self.val_accuracy(preds, moves)

        wrong = (preds != moves).sum().float()

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/errors', wrong, on_step=False, on_epoch=True, prog_bar=True, reduce_fx=torch.sum)

        # Pass the global checkersEvaluator
        if self.evaluator is not None:
            # checkersEvaluator expects (boards_tensor, pred_indices)
            illegal_count = self.evaluator(boards, preds)
            self.log('val/illegal', illegal_count, prog_bar=True)

    def predict_move(self, board_tensor: torch.Tensor) -> str:
        self.eval()
        with torch.no_grad():
            logits = self(board_tensor.unsqueeze(0))
            best_idx = torch.argmax(logits, dim=-1).item()
        if self.converter is None:
            raise ValueError("Funkcja konwertująca w modelu nie została podana!")
        return self.converter(best_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)