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
    
    comparatif = comparator.get_word(file_data["data"])
    
    if comparatif == "yes" :
            
        data.write_json(file_data["file"], file_data["data"], "D:/Niobium/database/yes.json", "yes")
    
    elif comparatif == "no" :
        
        data.write_json(file_data["file"], file_data["data"], "D:/Niobium/database/no.json", "no")

print("\nData processed !")