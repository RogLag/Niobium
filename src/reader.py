# ----------- #
#   Imports   #
# ----------- #

import numpy as np
from scipy.io import wavfile

# ----------- #
#   Fonction  #
# ----------- #

def read_wav(file):
    
    """
    Read a wav file and return fourier datas
    """
    try: 
        rate, aud_data = wavfile.read(file)
        
        fourier = np.fft.fft(aud_data)
        
        return {"data": fourier, "len": len(fourier), "rate": rate, "aud_data": aud_data, "file": file}
    
    except Exception as e:
        return None
