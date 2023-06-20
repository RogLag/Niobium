import numpy as np
import scipy.io.wavfile as wav
from tensorflow.keras.models import load_model
from model import preprocess_data

import warnings
warnings.filterwarnings("ignore")

# Charger le modèle entraîné
model = load_model('model.h5')

# Charger un fichier audio à tester
sample_rate1, audio1 = wav.read('Niobium/non1.wav')
sample_rate2, audio2 = wav.read('Niobium/oui2.wav')
sample_rate3, audio3 = wav.read('Niobium/oui3.wav')
sample_rate4, audio4 = wav.read('Niobium/non4.wav')
sample_rate5, audio5 = wav.read('Niobium/oui5.wav')
sample_rate6, audio6 = wav.read('Niobium/non6.wav')

# Prétraiter les données audio
mfccs1 = preprocess_data(audio1)
mfccs2 = preprocess_data(audio2)
mfccs3 = preprocess_data(audio3)
mfccs4 = preprocess_data(audio4)
mfccs5 = preprocess_data(audio5)
mfccs6 = preprocess_data(audio6)

# Ajouter une dimension supplémentaire pour correspondre à la forme d'entrée du modèle
mfccs1 = np.expand_dims(mfccs1, axis=-1)
mfccs2 = np.expand_dims(mfccs2, axis=-1)
mfccs3 = np.expand_dims(mfccs3, axis=-1)
mfccs4 = np.expand_dims(mfccs4, axis=-1)
mfccs5 = np.expand_dims(mfccs5, axis=-1)
mfccs6 = np.expand_dims(mfccs6, axis=-1)

# Faire une prédiction avec le modèle
prediction1 = model.predict(mfccs1)
prediction2 = model.predict(mfccs2)
prediction3 = model.predict(mfccs3)
prediction4 = model.predict(mfccs4)
prediction5 = model.predict(mfccs5)
prediction6 = model.predict(mfccs6)

# Afficher la prédiction (0 pour "non" et 1 pour "oui")

def moyenne_list(liste):
    total = 0
    for i in liste:
        total += i[0]
    return total/len(liste)

prediction1 = prediction1.tolist()
prediction2 = prediction2.tolist()
prediction3 = prediction3.tolist()
prediction4 = prediction4.tolist()
prediction5 = prediction5.tolist()
prediction6 = prediction6.tolist()

prediction1 = moyenne_list(prediction1)
prediction2 = moyenne_list(prediction2)
prediction3 = moyenne_list(prediction3)
prediction4 = moyenne_list(prediction4)
prediction5 = moyenne_list(prediction5)
prediction6 = moyenne_list(prediction6)

print("prediction1 : ", prediction1)
print("prediction2 : ", prediction2)
print("prediction3 : ", prediction3)
print("prediction4 : ", prediction4)
print("prediction5 : ", prediction5)
print("prediction6 : ", prediction6)

#on enleve 0.5% pour avoir une marge d'erreur


if  prediction1 > 0.497201:
    print("prediction1 : La réponse est: oui")
else:
    print("prediction1 : La réponse est: non")
    
if  prediction2 > 0.497201:
    print("prediction2 : La réponse est: oui")
else:
    print("prediction2 : La réponse est: non")
    
if  prediction3 > 0.497201:
    print("prediction3 : La réponse est: oui")
else:
    print("prediction3 : La réponse est: non")
    
if  prediction4 > 0.497201:
    print("prediction4 : La réponse est: oui")
else:
    print("prediction4 : La réponse est: non")
    
if  prediction5 > 0.497201:
    print("prediction5 : La réponse est: oui")
else:
    print("prediction5 : La réponse est: non")
    
if  prediction6 > 0.497201:
    print("prediction6 : La réponse est: oui")
else:
    print("prediction6 : La réponse est: non")