# ----------- #
#   Imports   #
# ----------- #

import json
import numpy as np

# ----------- #
#    Class    #
# ----------- #

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ----------- #
#   Fonction  #
# ----------- #

def json_serializable(obj):
    if isinstance(obj):
        return obj.__str__()

def read_json(path):
        
    """
    Read a json file
    """
    
    file = open(path, "r")
    return json.load(file)


def write_json(name, data, path, type2):
    
    """
    Write a json file
    """
    
    database = read_json(path)
        
    database[name] = str(data.tolist())
    
    with open(f'D:/Niobium/database/{type2}.json', 'w') as mon_fichier:
	    json.dump(database, mon_fichier)
