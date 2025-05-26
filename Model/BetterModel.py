import torch
import torch.nn as nn
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) * (torch.log(torch.tensor(10000.0)) / d_model))
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


class ChessTransformer(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 64,
        num_moves: int = 4096,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lr: float = 3e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Token embedding and positional encoding
        self.embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        x = self.embedding(x)               # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)            # plus positional encoding

        # Causal mask for self-attention
        device = x.device
        # attn_mask shape: (seq_len, seq_len), float with -inf on upper triangle
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Pool across sequence and classify
        pooled = x.mean(dim=1)             # (batch_size, d_model)
        logits = self.fc(pooled)           # (batch_size, num_moves)
        return logits

    def training_step(self, batch, batch_idx):
        boards, moves = batch
        logits = self(boards)
        loss = self.loss_fn(logits, moves)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def predict_move(self, board_tensor: torch.Tensor) -> str:
        self.eval()
        with torch.no_grad():
            logits = self(board_tensor.unsqueeze(0))
            best_idx = torch.argmax(logits, dim=-1).item()
        start, end = self.index_to_notation(best_idx)
        return f"{start}{end}"

    def index_to_notation(self, index: int) -> (str, str):
        start_idx = index // 64
        end_idx = index % 64
        return self._idx_to_coord(start_idx), self._idx_to_coord(end_idx)

    @staticmethod
    def _idx_to_coord(idx: int) -> str:
        row = 8 - (idx // 8)
        col = chr((idx % 8) + ord('a'))
        return f"{col}{row}"

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @property
    def total_parameters(self) -> int:
        #liczy liczbę parametrów:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#
# model = ChessTransformer(d_model=256, num_heads=4, num_layers=4)
# print(f"Total parameters: {model.total_parameters}")
