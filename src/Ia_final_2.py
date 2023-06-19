import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import wavfile

file = open("D:/Virgile/audio/non.3vgpd6n4.ingestion-7b68fffd8-smmnb.s48.wav", "rb")
file2 = open("D:/Virgile/audio/oui.3vgp484f.ingestion-7b68fffd8-mmz2v.s56.wav", "rb")

rate, aud_data = scipy.io.wavfile.read(file)
rate2, aud_data2 = scipy.io.wavfile.read(file2)


channel_1 = aud_data[:]
channel_2 = aud_data2[:]


fourier = np.fft.fft(channel_1)
fourier2 = np.fft.fft(channel_2)

print("fourier 1 : ",fourier," ",len(fourier))

print("-----------------------------------------------")

print("fourier 2 : ",fourier2," ",len(fourier2))