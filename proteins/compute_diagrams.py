import numpy as np
import os
import sys
import pickle


from ripser import ripser
from persim import plot_diagrams

diagrams = []

j = 0
for file in os.listdir('./SCOP40mini/'):
    filename = os.fsdecode(file)
        
    open_file = open('./SCOP40mini/'+filename)
    lines = open_file.read().splitlines()
    
    coordinates = []
    
    for i in range(12, len(lines)-3):
        line = lines[i]

        if line.split()[0]=='ATOM':
            x = float(line[32:38])
            y = float(line[38:46])
            z = float(line[46:54])
                        
            coordinates.append([x, y, x])
            
        
    coordinates = np.array(coordinates)
    
    if len(coordinates)>1000:
        d = ripser(coordinates, maxdim=1, do_cocycles=False, n_perm=1000)['dgms']
    else:
    	d = ripser(coordinates, maxdim=1, do_cocycles=False)['dgms']
    
    name = filename[:-4]
    
    temp = {}

    temp['name'] = name
    temp['h0'] = d[0]
    temp['h1'] = d[1]

    diagrams.append(temp)

    print(j)
    j += 1
        
# Save with pickle
with open('diagrams.pickle', 'wb') as handle:
    pickle.dump(diagrams, handle, protocol=pickle.HIGHEST_PROTOCOL)