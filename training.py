import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import time
import testing
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

def multiple(): 

    mfcc_data = []

    labels = []

    for file in glob.glob("audio/*.wav"):
        if file.endswith('.wav'):
            rate, data = wav.read(file)
            mfcc_features = mfcc(data, rate)
            mfcc_data.append(mfcc_features)
            if gettype(file) == "oui":
                labels.append(np.ones(len(mfcc_features)))
            else:
                labels.append(np.zeros(len(mfcc_features)))
            
    # Concaténer les données et les labels
    data = np.concatenate(mfcc_data)
    labels = np.concatenate(labels)
    
    return data, labels
        
def single():
    rate_oui, data_oui = wav.read("witness/oui.wav")
    rate_oui2, data_oui2 = wav.read("witness/oui2.wav")
    rate_non, data_non = wav.read("witness/non.wav")
    rate_non2, data_non2 = wav.read("witness/non2.wav")

    mfcc_oui = mfcc(data_oui, rate_oui)
    mfcc_non = mfcc(data_non, rate_non)
    mfcc_oui2 = mfcc(data_oui2, rate_oui2)
    mfcc_non2 = mfcc(data_non2, rate_non2)

    # Créer les labels correspondants
    labels_oui = np.ones(len(mfcc_oui))
    labels_non = np.zeros(len(mfcc_non))
    labels_oui2 = np.ones(len(mfcc_oui2))
    labels_non2 = np.zeros(len(mfcc_non2))

    # Concaténer les données et les labels
    data = np.concatenate((mfcc_oui, mfcc_non, mfcc_oui2, mfcc_non2))
    labels = np.concatenate((labels_oui, labels_non, labels_oui2, labels_non2))
    
    return data, labels

data, labels = single()

# Convertir les données en tenseurs PyTorch
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).long()

# Créer un DataLoader pour l'entraînement
dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# charger le modèle
model = Model()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
def train():
    num_epochs = 100
    print("Training...")
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
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), "model.pt")
    
while True:
    print("\n")
    train()
    testing.testing_multiple()
    testing.testing_single()
    time.sleep(3)