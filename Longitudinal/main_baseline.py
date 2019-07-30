'''
Baseline model training and testing
'''

from models import deep_cnn

import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, CSVLogger
from keras import backend as K

from customs import CustomCallback, acc_likelihood, acc_threshold

import sys, os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or other if you have more;
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes the tensorflow initial information

DIR = 'numbers' # unzipped train and test data

BATCH = 16
EPOCHS = 50

LABELS = 'one two three four five six seven eight nine'.split()
NUM_CLASSES = len(LABELS)

#==============================================================================
# Prepare data      
# #==============================================================================
# dsGen = DatasetGenerator(label_set=LABELS) 
# # # Load DataFrame with paths/labels for training and validation data 
# # # and paths for testing data 
# df = dsGen.load_data(DIR)

# dsGen.apply_train_test_split(test_size=0.0, random_state=2018)
# dsGen.apply_train_val_split(val_size=0.2, random_state=2018)

# dsGen.build_dataset(mode='train')
# dsGen.build_dataset(mode='val')

# sys.exit()

x_train = np.load('train_batches.npy')
y_train = np.load('train_targets.npy')
x_test = np.load('val_batches.npy')
y_test = np.load('val_targets.npy')

maxf = np.max(np.array([np.max(np.abs(x_train)),np.max(np.abs(x_test))]))

x_train = x_train/maxf
x_test = x_test/maxf

INPUT_SHAPE = x_train.shape[1:]

nsample = x_train.shape[0]

print(INPUT_SHAPE)


ssplit = [128, 512, 1024,  nsample/10, nsample/5, nsample/2, nsample]

nsplit = len(ssplit) 

reps = 32

folder = './LogsB/longitudinal/'
for k in range(reps):

	if not os.path.exists(folder+str(k)):
		os.makedirs(folder+str(k))

	for i in range(nsplit):
		a=k%(nsample/ssplit[i])
			
		x_split = x_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		y_split = y_train[(ssplit[i]*a):(ssplit[i]*(a+1))]

		print(ssplit)

		csv_logger = CSVLogger(folder+str(k)+'/log_'+"{:03d}".format(i)+'.csv')
		custom_callback = CustomCallback(k,i,x_test,y_test,folder)
		callbacks = [custom_callback,csv_logger]

		batch_norm=True
		if (i<5):
			batch_norm=False

		model = deep_cnn(INPUT_SHAPE, NUM_CLASSES,batch_norm=batch_norm)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics={"class_output": ['accuracy',acc_likelihood,acc_threshold]})

		history = model.fit([x_split], [y_split],
							batch_size=BATCH,
							epochs=EPOCHS,
							callbacks=callbacks,
							shuffle=True,
							verbose=0,
							validation_data=([x_test], [y_test]))

		K.clear_session()
