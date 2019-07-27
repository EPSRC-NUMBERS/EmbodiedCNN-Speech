'''
Baseline model training and testing - with Zipfian distribution
'''

from models import deep_cnn

import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, CSVLogger,ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
from customs import CustomCallback, acc_likelihood, acc_threshold

import sys, os

DIR = 'numbers' # unzipped train and test data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or other if you have more;
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes the tensorflow initial information 

BATCH = 16
EPOCHS = 50

LABELS = 'one two three four five six seven eight nine'.split()
NUM_CLASSES = 9
NUM_OUTPUTS = 9

x_train = np.load('train_batches.npy')
y_train = np.load('train_targets.npy')
x_test = np.load('val_batches.npy')
y_test = np.load('val_targets.npy')

maxf = np.max(np.array([np.max(np.abs(x_train)),np.max(np.abs(x_test))])) #finds the maximum value of the entire database

x_train = np.load('zip_train.npy')
y_train = np.load('zip_targets.npy')
pre_train = np.load('zip_pretrain.npy')
pre_targets = np.load('zip_pretargets.npy')

# scale in [0.1]
x_train = np.abs(x_train)/maxf
x_test = np.abs(x_test)/maxf
pre_train = np.abs(pre_train)/maxf

INPUT_SHAPE = x_train.shape[1:]

nsample = x_train.shape[0]

print('nsample = ',nsample)

ssplit = [(nsample/4),nsample/2,nsample]
divs = [4,2,1]

nsplit = len(ssplit) 

#==============================================================================
# Train
#==============================================================================  
# 

reps = 25

pre_epochs = np.zeros((nsplit,reps))
folder = './LogsB/zipfian/'
for k in range(reps):

	if not os.path.exists(folder+str(k)):
		os.makedirs(folder+str(k))
		
	for i in range(nsplit):
		a=k%divs[i]

		x_split = x_train[a::divs[i]]
		y_split = y_train[a::divs[i]]
		
		custom_callback = CustomCallback(k,i,x_test,y_test,folder)
		csv_logger = CSVLogger(folder+str(k)+'/log_'+"{:03d}".format(i)+'.csv')
		callbacks = [custom_callback,csv_logger]

		model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics={"class_output": ['accuracy',acc_likelihood,acc_threshold]})

		plot_model(model,to_file=folder+'baseline.png')

		history = model.fit([x_split], [y_split],
							batch_size=BATCH,
							epochs=EPOCHS,
							callbacks=callbacks,
							shuffle=True,
							verbose=0,
							validation_data=([x_test], [y_test]))

		K.clear_session()
