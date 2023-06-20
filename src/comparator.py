# Ici, on compare les données recu de fourier pour savoir si c'est un "oui" ou un "non" ou un "unknown"

import reader
import numpy as np

def get_word(data_fourier):
        
        """
        Cette fonction ce base sur des témoins de "oui" et de "non" pour déterminer si le mot est un "oui" ou un "non"
        """
        
        # Get the data of the witness
        yes = reader.read_wav("temoin/oui.3vgp65o0.ingestion-7b68fffd8-4vtgd.s1.wav")
        no = reader.read_wav("temoin/non.3vgpacp1.ingestion-7b68fffd8-mmz2v.s1.wav")
        
        # Get the data of the file
        file = data_fourier
        
        # Get the data of the witness
        yes_witness = np.array(yes["data"])
        no_witness = np.array(no["data"])
        
        # Get the data of the file
        file_data = np.array(file)
        
        # Get the difference between the witness and the file
        yes_diff = np.abs(np.subtract(yes_witness, file_data))
        no_diff = np.abs(np.subtract(no_witness, file_data))
        
        # Get the sum of the difference
        yes_sum = np.sum(yes_diff)
        no_sum = np.sum(no_diff)
        
        # Get the type of the word
        if yes_sum < no_sum :
            
            return "yes"
        
        elif yes_sum > no_sum :
            
            return "no"
        
        else :
            
            return "unknown"