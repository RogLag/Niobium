import drive
import librosa
import pre_traitement

print("\nGetting files... \n")

files = drive.get_files()

print("\nFiles got !")

for file in files:
    
    audio, sr = librosa.load(f"../audio/{file}", sr=None)
    preprocessed_audio = pre_traitement.preprocess_audio(audio)