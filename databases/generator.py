'''
Database Generator
'''

from dataset import DatasetGenerator

import sys, os


DIR = 'numbers' # unzipped train and test data


LABELS = 'one two three four five six seven eight nine'.split()
NUM_CLASSES = len(LABELS)

#==============================================================================
# Prepare data      
#==============================================================================
dsGen = DatasetGenerator(label_set=LABELS) 
# Load DataFrame with paths/labels for training and validation data 
# and paths for testing data 
df = dsGen.load_data(DIR)

dsGen.apply_train_test_split(test_size=0.0, random_state=2018)
dsGen.apply_train_val_split(val_size=0.2, random_state=2018)

dsGen.build_dataset(mode='train')
dsGen.build_dataset(mode='val')

sys.exit()