import torch
import torch.nn as nn
import lightning as l


class PositionEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=64):
        super().__init__()

        pe = torch.zeros(max_len, d_model) #tworzymy macierz o wymiarach max_len na d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #tworzymy sekwencje floatow od 0 do max_len
        #unsqueeze(1) - zamienia sekwencje na macierz kolumnowa

        embedding_index = torch.arange(0, d_model, step=2).unsqueeze(0) #tworzymy sekwencje floatow od 0 do d_model i jest to macierz wierszowa
        #step = 2; czyli będzie 0,2,4,.... chodzi o to, by bylo to 2i jak we wzorze

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term) #dla parzystych 'i' dajemy sin
        pe[:, 1::2] = torch.cos(position * div_term) #dla nieparzystych - cos

        self.register_buffer('pe', pe) #upewniamy się, że pe zostaje przydzielone do GPU (jesli istnieje)
        # TRZEBA DOCZYTAĆ

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]


class Attention(nn.Module):
    def __init__(self, d_model=512, max_len=64):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model) #nn.Linear() stworzy Weight matrix i zajmie sie cala matma
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.row_dim = 0
        self.col_dim = 1
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))


    def forward(self, encoding_for_q, encoding_for_k, encoding_for_v, mask=None):
        q = self.W_q(encoding_for_q)
        k = self.W_k(encoding_for_k)
        v = self.W_v(encoding_for_v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)

        return output


class ChessTransformer(l.LightningModule):
    def __init__(self, d_model=512, max_len=64, num_moves=4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        self.position_encoding = PositionEncoding()
        self.attention = Attention()
        self.fc = nn.Linear(in_features=d_model, out_features=num_moves)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.position_encoding(x)

        mask = torch.tril(torch.ones((x.size(dim=0), x.size(dim=0))))
        mask = mask == 0
        attention = self.attention(x, x, x, mask)
        residual_connection_values = x + attention
        logits_seq = self.fc(residual_connection_values)
        logits = logits_seq.mean(dim=1)
        return logits  #logits



    def predict_move(self, board_tensor):
        with torch.no_grad():
            logits = self.forward(board_tensor)  # Przewidywanie logitów dla możliwych ruchów
            #best_move_index = torch.argmax(logits, dim=-1).cpu().numpy()[0] #nie wiem czy nie zamienić na torch.argmax(logtis, dim=-1).item()
            best_move_idx = torch.argmax(logits, dim=-1).item()

        start, end = self.index_to_notation(best_move_idx)
        return start + end

    def training_step(self, batch, batch_idx):
        boards, moves = batch
        logits = self.forward(boards)
        loss = self.loss(logits, moves)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def index_to_notation(self, index):
        # Zamiana indeksu na współrzędne (0-36)
        start_index = index // 64  # Pole początkowe
        end_index = index % 64  # Pole docelowe

        # Zamiana współrzędnych na notację "jakąś" (np. 0 -> 'a1', 63 -> 'h8')
        start = self.index_to_coordinate(start_index)
        end = self.index_to_coordinate(end_index)

        return start, end


    def index_to_coordinate(self, index):
        row = 8 - (index // 8)  # Liczba wiersza (od 1 do 8)
        col = chr(index % 8 + ord('a'))  # Kolumna (od 'a' do 'h')
        return col + str(row)

