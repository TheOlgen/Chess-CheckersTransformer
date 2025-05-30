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
            max_len: int = 101,  # Max board tensor length (100 board + 1 color)
            num_moves: int = 2500,  # Max possible move indices (50 * 50 = 2500)
            num_heads: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            lr: float = 3e-4,
            evaluator=None  # This will be checkersEvaluator
    ):
        super().__init__()
        self.save_hyperparameters()

        # Token embedding and positional encoding
        # num_embeddings: Max value in your board tensor (0-4 for pieces/empty) + 1 for active color.
        # Let's say max value is 4 (for B king), and 1 for white to move. So 5.
        # It should be max_val_in_tensor + 1. Here, 0,1,2,3,4 for pieces and 0,1 for color.
        # The embedding layer will take the values (0,1,2,3,4) for pieces and (0,1) for color.
        # If your tensor contains values up to 4, then num_embeddings should be 5.
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=d_model)  # Pieces are 0-4
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.evaluator = evaluator

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final classification head
        self.fc = nn.Linear(d_model, num_moves)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_moves)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len) -> (batch_size, 101) where 101 is 100 board + 1 color
        batch_size, seq_len = x.size()

        # Ensure x is used as token indices for embedding.
        # x will contain values like 0, 1, 2, 3, 4.
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)  # plus positional encoding

        # Causal mask for self-attention - usually not used for classification of entire sequences
        # Causal mask makes sense for predicting *next token* given previous tokens.
        # For classifying the entire board state to a move, a causal mask might not be appropriate
        # unless you interpret the input sequence as a sequential input, which is unusual for board games.
        # If you intend for the model to "see" the entire board at once, remove or adjust this mask.
        # For a standard Transformer Encoder setup (which this looks like), no causal mask is needed.
        # If it's a Decoder-like setup, then keep it. Given your current architecture,
        # where you pool at the end, it suggests an Encoder. Let's remove it for now for board classification.
        # If you want to keep it, ensure it's (seq_len, seq_len) and proper type.

        # attn_mask is typically for masking future tokens or padding
        # For a board representation, all input features are available.
        # If you want to mask padding (e.g. if seq_len < max_len), you need key_padding_mask.
        # Assuming seq_len is always max_len (101), no mask is strictly necessary for basic encoder.
        attn_mask = None
        key_padding_mask = None

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Pool across sequence and classify
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

        wrong = (preds != moves).sum()

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/errors', wrong, on_step=False, on_epoch=True, prog_bar=True)

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
        # You need to import index_to_move here or pass it
        from CheckersDataset import index_to_move  # Example import
        return index_to_move(best_idx)

    # _idx_to_coord and index_to_notation are for chess, remove them or adapt
    # def index_to_notation(self, index: int) -> (str, str): ...
    # @staticmethod
    # def _idx_to_coord(idx: int) -> str: ...

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)