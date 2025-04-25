import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from states_loader import StatesLoader


class TicTacToeModel(nn.Module):
    def __init__(self, board_size = 3):
        super(TicTacToeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)
        self.relu = nn.ReLU()
        conv_output_size = board_size - 1  # dla board_size=3: 3-2+1 = 2
        self.fc1 = nn.Linear(32 * conv_output_size * conv_output_size, 64)
        self.fc2 = nn.Linear(64, board_size * board_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TicTacToeDataset(Dataset):
    def __init__(self, data, board_size=3):
        self.data = [entry for entry in data if entry["best_move"] is not None]
        self.board_size = board_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        board_before = np.array(entry["board"]).astype(np.float32)
        board_tensor = torch.tensor(board_before).unsqueeze(0)  # kształt: [1, board_size, board_size]

        best_move = entry["best_move"]
        row = best_move[0]
        col = best_move[1]

        move_index = int(row) * self.board_size + int(col)
        target = torch.tensor(move_index).long()
        return board_tensor, target


board_size = 3
batch_size = 32

X = torch.randn(1000, 1, board_size, board_size)
y = torch.randint(0, board_size * board_size, (1000,))

states_with_labels_3x3 = StatesLoader.load_processed_states("states_with_labels_3x3.pkl")
dataset = TicTacToeDataset(states_with_labels_3x3)
#train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = TicTacToeModel(board_size=board_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()  #model w tryb ewaluacji
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy*100:.2f}%")