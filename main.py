import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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

def gettype(file):
    if 'oui' in file:
        return "oui"
    else:
        return "non"

# Charger et prétraiter les fichiers audio pour l'entraînement

mfcc_data = []

labels = []

compteur = 0

for file in glob.glob("audio/*.wav"):
    compteur += 1
        
    type = "non"
        
    if compteur > len(glob.glob("audio/*.wav"))//2:
        type = "oui"
        
    if file.endswith('.wav'):
        rate, data = wav.read(file)
        mfcc_features = mfcc(data, rate)
        mfcc_data.append(mfcc_features)
        if type == "oui":
            labels.append(np.ones(len(mfcc_features)))
        else:
            labels.append(np.zeros(len(mfcc_features)))
        
# Concaténer les données et les labels
data = np.concatenate(mfcc_data)
labels = np.concatenate(labels)

# Convertir les données en tenseurs PyTorch
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).long()

# Créer un DataLoader pour l'entraînement
dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Créer le modèle
model = Model()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 10
print("\nCréation du modèle...")
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Enregistrer le modèle
torch.save(model.state_dict(), "model.pt")
