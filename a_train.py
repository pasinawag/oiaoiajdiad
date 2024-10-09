import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load the data
data_dict = torch.load('./data.pt')

data = data_dict['data']
labels = data_dict['labels']

# Create a dataset and dataloaders
dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define a simple neural network model
class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.fc1 = nn.Linear(42, 128)  # 42 because there are 21 landmarks (x,y) coordinates
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # 5 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model, loss function, and optimizer
model = SignLanguageModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
epochs = 500
for epoch in range(epochs):
    running_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
