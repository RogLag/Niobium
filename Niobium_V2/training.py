import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Définir l'architecture du modèle
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(13, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Charger et prétraiter les fichiers audio pour l'entraînement
rate_oui, data_oui = wav.read("Niobium_V2/oui.wav")
rate_non, data_non = wav.read("Niobium_V2/non.wav")

mfcc_oui = mfcc(data_oui, rate_oui)
mfcc_non = mfcc(data_non, rate_non)

# Créer les labels correspondants
labels_oui = np.ones(len(mfcc_oui))
labels_non = np.zeros(len(mfcc_non))

# Concaténer les données et les labels
data = np.concatenate((mfcc_oui, mfcc_non))
labels = np.concatenate((labels_oui, labels_non))

# Convertir les données en tenseurs PyTorch
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).long()

# Créer un DataLoader pour l'entraînement
dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# charger le modèle
model = Model()
model.load_state_dict(torch.load("Niobium_V2/model.pt"))
model.eval()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Enregistrer le modèle
torch.save(model.state_dict(), "Niobium_V2/model.pt")
