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


def write_json(data, path, type2):
    
    """
    Write a json file
    """
    
    database = read_json(path)
        
    database.append(data.tolist())
    
    print(database[1:])
    print(type(database))
    
    json.dumps(database)
