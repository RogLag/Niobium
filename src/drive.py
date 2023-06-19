# ----------- #
#   Imports   #
# ----------- #

import glob
from tqdm import tqdm

# ----------- #
#   Fonction  #
# ----------- #

def get_files(root = "D:/Virgile/audio/"):
    
    """
    Get all files in the directory
    """

    files = []
    
    for file in tqdm(glob.glob("audio/*.wav")):
        files.append(file[6:])
            
    return files