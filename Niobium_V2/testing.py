import torch
import torch.nn as nn
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import glob

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

# Charger le modèle
model = Model()
model.load_state_dict(torch.load("Niobium_V2/model.pt"))
model.eval()

# Liste des fichiers à tester

fichiers_test = []

reussite = 0
echec = 0

for file in glob.glob("audio/*.wav"):
    if file.endswith('.wav'):
        fichiers_test.append(file)

# Parcourir les fichiers de test
for fichier in fichiers_test:
    # Prétraiter le fichier audio pour le tester
    rate, data = wav.read(fichier)
    mfcc_features = mfcc(data, rate)
    input_data = torch.from_numpy(mfcc_features).float()

    # Ajouter une dimension pour correspondre à la dimension de batch attendue par le modèle
    input_data = input_data.unsqueeze(0)

    # Effectuer la prédiction avec le modèle chargé
    with torch.no_grad():
        output = model(input_data)
    predicted_label = torch.argmax(output, dim=1)
    
    print(predicted_label)

    # Interpréter la prédiction
    if predicted_label[0][0].item() < 50:
        print(f"Le mot 'oui' a été détecté dans le fichier {fichier}.")
        if gettype(file) == "oui":
                reussite += 1
        else:
                echec += 1
    else:
        print(f"Le mot 'non' a été détecté dans le fichier {fichier}.")
        if gettype(file) == "non":
                reussite += 1
        else:
                echec += 1

# Afficher le pourcentage de réussite
print(f"Pourcentage de réussite: {reussite/len(fichiers_test)*100}%")
print(f"Nombre de réussites: {reussite}")
print(f"Nombre d'échecs: {echec}")