import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
import os
import glob
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Chargement des données audio
def load_data(data_dir):
    # Charger les fichiers audio dans un tableau numpy
    data = []
    labels = []
    # Parcourir les fichiers audio dans le dossier "data"
    for file in glob.glob("audio/*.wav"):
        if file.endswith('.wav'):
            # Lire le fichier audio et l'ajouter au tableau de données
            sample_rate, audio = wav.read("audio/"+file[6:])
            data.append(audio)
            # Ajouter l'étiquette "oui" ou "non" en fonction du nom du fichier
            if 'oui' in file:
                labels.append(1)
            else:
                labels.append(0)
    return np.array(data), np.array(labels)

# Prétraitement des données audio
def preprocess_data(data):
    # Normaliser les données audio
    data = data / 32768.0
    # Appliquer une fenêtre de Hamming aux données audio
    window = np.hamming(1024)
    data = np.array([np.convolve(window, d)[:1024] for d in data])
    # Ajouter une dimension pour la couleur (1 pour les fichiers audio mono)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    return data

# Définition du modèle CNN-LSTM
def define_model(shape):
    # Définir le modèle
    model = Sequential([
        # Couche LSTM avec 64 neurones et une activation de tangente hyperbolique
        LSTM(64, activation='tanh', input_shape=shape),
        # Couche de dropout pour la régularisation
        Dropout(0.2),
        # Couche entièrement connectée avec 1 neurone et une fonction d'activation sigmoïde pour la classification binaire
        Dense(1, activation='sigmoid')
    ])

    # Compiler le modèle avec une fonction de perte binaire_crossentropy et un optimiseur Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entraînement du modèle
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    print('Test Accuracy: %.3f' % acc)

if __name__ == '__main__':
    # Chargement des données audio
    data_dir = 'data'
    data, labels = load_data(data_dir)
    
    # Prétraitement des données audio
    preprocessed_data = preprocess_data(data)
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=0.2)
    
    # Définition du modèle CNN-LSTM
    input_shape = preprocessed_data[0].shape
    model = define_model(input_shape)
    
    # Entraînement du modèle
    train_model(model, X_train, y_train, X_test, y_test)
    
    # Évaluation du modèle
    evaluate_model(model, X_test, y_test)
    
    # Sauvegarde du modèle
    model.save('model.h5')