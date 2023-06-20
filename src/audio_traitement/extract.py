import numpy as np
import librosa

def split_audio_into_frames(audio, sample_rate, frame_duration):
    frame_size = int(frame_duration * sample_rate)  # Taille de la trame en échantillons
    hop_size = int(frame_duration * sample_rate)  # Taille du pas entre les trames en échantillons

    total_samples = len(audio)
    total_frames = int(np.ceil((total_samples - frame_size) / hop_size)) + 1

    frames = np.zeros((total_frames, frame_size))

    for i in range(total_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i] = audio[start:end]

    return frames

def extract_features(audio, sample_rate):
    # Extraction des coefficients MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    return mfcc