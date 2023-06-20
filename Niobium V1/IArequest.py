import numpy as np
import scipy.io.wavfile as wav
from tensorflow.keras.models import load_model
from model import preprocess_data
import glob

import warnings
warnings.filterwarnings("ignore")

def moyenne_list(liste):
            total = 0
            for i in liste:
                total += i[0]
            return total/len(liste)
        
def gettype(file):
    if 'oui' in file:
        return "oui"
    else:
        return "non"
    
# Charger le modèle entraîné
model = load_model('model.h5')

# Charger tous les fichiers audio à tester

set = 0

for file in glob.glob("audio/oui*.wav"):
    if file.endswith('.wav'):
        # Lire le fichier audio et l'ajouter au tableau de données
        sample_rate, audio = wav.read("audio/"+file[6:])
        
        # Prétraiter les données audio
        mfccs = preprocess_data(audio)
        
        # Ajouter une dimension supplémentaire pour correspondre à la forme d'entrée du modèle
        mfccs = np.expand_dims(mfccs, axis=-1)
        
        # Faire une prédiction avec le modèle
        prediction = model.predict(mfccs)
        
        # Afficher la prédiction (0 pour "non" et 1 pour "oui")
        prediction = prediction.tolist()
        
        prediction = moyenne_list(prediction)
        
        print(f"prediction {gettype(file)} : ", prediction)
        
        if  prediction > 0.497201:
            print(f"prediction {gettype(file)} : La réponse est: oui")
            if gettype(file) == "oui":
                set += 1
        else:
            print(f"prediction {gettype(file)} : La réponse est: non")
            if gettype(file) == "non":
                set += 1

print(f"Le taux de réussite est de {set/len(glob.glob('audio/*.wav'))*100}%")
            