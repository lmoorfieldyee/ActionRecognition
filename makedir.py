"""
general purpose functions go in utils.py file
"""

import numpy as np
import os

# Path for exported data, numpy arrays
path = './Raw_Data'
person = 'William_desk' + '_Data'
if os.path.isdir(path)==False:
    os.makedirs(path)

DATA_PATH = os.path.join(path, person)

# Actions we want to detect
actions = np.array(['wave', 'kiss', 'finger', 'salute', 'heart', 'idle'])
#actions = ['idle']
for action in actions:
    if os.path.isdir(os.path.join(DATA_PATH, action)) == False:
        os.makedirs(os.path.join(DATA_PATH, action))

# Thirty videos worth of data per class
no_sequences = 30

# Folder start
start_folder = 30

for action in actions:
    if os.path.isdir(os.path.join(DATA_PATH, action)):
        dirmax = len(os.listdir(os.path.join(DATA_PATH, action)))
        for sequence in range(0,no_sequences):
            print(action, dirmax+sequence)
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
            except:
                pass