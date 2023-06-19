# Ici, on compare les données recu de fourier pour savoir si c'est un "oui" ou un "non" ou un "unknown"

def comparator(fourier, yes_witness, no_witness, accuracy="75%"):
        
        """
        Compare fourier datas with yes and no witness with 75% of accuracy
        """
        
        yes = 0
        no = 0
        
        for i in range(len(fourier)):
            
            if abs(fourier[i]) > abs(yes_witness[i]):
                
                yes += 1
                
            elif abs(fourier[i]) < abs(no_witness[i]):
                
                no += 1
                
        if yes > no:
            
            return {"type":"yes", "data": fourier, "len": len(fourier), "accuracy": yes/(yes+no)}
        
        elif no > yes:
            
            return {"type":"no", "data": fourier, "len": len(fourier), "accuracy": no/(yes+no)}
        
        else:
            
            return {"type":"unknown", "data": fourier, "len": len(fourier), "accuracy": accuracy}