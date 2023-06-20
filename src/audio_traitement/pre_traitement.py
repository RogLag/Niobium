import librosa

def preprocess_audio(audio):
    # Réduction du bruit
    audio = librosa.effects.trim(audio)[0]

    # Normalisation du volume
    normalized_audio = librosa.util.normalize(audio)

    return normalized_audio