# ----------- #
#   Imports   #
# ----------- #

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import drive
import reader
import comparator
import data

# ------------ #
# Body of code #
# ------------ #

#print(data.read_json("D:/Niobium/database/data.json"))

# Get all the files in the directory

print("\nGetting files... \n")

files = drive.get_files()

print("\nFiles got !")

# Read all the files

print("\nReading files... \n")

all_data = []

for file in tqdm(files):
    
    all_data.append(reader.read_wav("audio/"+file))
    
print("\nAll files read !")

# Processing data

print("\nProcessing data... \n")

for file_data in tqdm(all_data):
    
    comparatif = comparator.comparator(file_data["data"], reader.read_wav("temoin/oui.3vgp65o0.ingestion-7b68fffd8-4vtgd.s1.wav")["data"], reader.read_wav("temoin/non.3vgpacp1.ingestion-7b68fffd8-mmz2v.s1.wav")["data"])
    
    if comparatif["type"] == "yes" :
            
        data.write_json(comparatif["data"], "D:/Niobium/database/yes.json", "yes")
    
    elif comparatif["type"] == "no" :
        
        data.write_json(comparatif["data"], "D:/Niobium/database/no.json", "no")

print("\nData processed !")