import torch
import torch.nn as nn
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

def testing_multiple(printing = False):
    # Charger le modèle
    model = Model()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # Liste des fichiers à tester

    fichiers_test = []

    reussite = 0
    echec = 0

    for file in glob.glob("audio/*.wav"):
        if file.endswith('.wav'):
            fichiers_test.append(file)

    # Parcourir les fichiers de test
    print('\ntesting...')
    compteur = -1
    for fichier in tqdm(fichiers_test):
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
        
        compteur += 1
        
        type = "non"
        
        if compteur > len(fichiers_test)//2:
            type = "oui"

        # Interpréter la prédiction
        if predicted_label[0][0].item() < 50:
            if printing:
                print(f"Le mot 'oui' a été détecté dans le fichier {fichier} avec {predicted_label}.")
            if type == "oui":
                    reussite += 1
                    if printing:
                        print("Réussite")
            else:
                    echec += 1
                    if printing:
                        print("Echec")
        else:
            if printing:
                print(f"Le mot 'non' a été détecté dans le fichier {fichier} avec {predicted_label}.")
            if type == "non":
                    reussite += 1
                    if printing:
                        print("Réussite")
            else:
                    echec += 1
                    if printing:
                        print("Echec")
        
    # Afficher le pourcentage de réussite
    print(f"Pourcentage de réussite: {round(reussite/len(fichiers_test)*100, 2)}%")
    print(f"Nombre de réussites: {reussite}")
    print(f"Nombre d'échecs: {echec}")

def testing_single(printing = False):
    # Charger le modèle
    model = Model()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # Liste des fichiers à tester

    fichiers_test = []

    reussite = 0
    echec = 0

    for file in glob.glob("audio_test/*.wav"):
        if file.endswith('.wav'):
            fichiers_test.append(file)

    # Parcourir les fichiers de test
    print('\ntesting...')
    compteur = -1
    for fichier in tqdm(fichiers_test):
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

        compteur += 1
        
        type = "non"
        
        if compteur > len(fichiers_test)//2:
            type = "oui"

        # print(predicted_label)

        # Interpréter la prédiction
        if predicted_label[0][0].item() < 50:
            if printing:
                print(f"Le mot 'oui' a été détecté dans le fichier {fichier} avec {predicted_label}.")
            if type == "oui":
                    reussite += 1
                    if printing:
                        print("Réussite")
            else:
                    echec += 1
                    if printing:
                        print("Echec")
        else:
            if printing:
                print(f"Le mot 'non' a été détecté dans le fichier {fichier} avec {predicted_label}.")
            if type == "non":
                    reussite += 1
                    if printing:
                        print("Réussite")
            else:
                    echec += 1
                    if printing:
                        print("Echec")

    # Afficher le pourcentage de réussite
    print(f"Pourcentage de réussite: {round(reussite/len(fichiers_test)*100, 2)}%")
    print(f"Nombre de réussites: {reussite}")
    print(f"Nombre d'échecs: {echec}")
    
if __name__ == "__main__":
    testing_multiple()
    testing_single()