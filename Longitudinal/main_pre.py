'''
Baseline with pre-training
'''


from models import emb_cnn_pre,emb_cnn_full

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

BATCH = 32
EPOCHS = 25

LABELS = 'one two three four five six seven eight nine'.split()
NUM_CLASSES = 9
NUM_OUTPUTS = 16

x_train = np.load('train_batches.npy')
y_train = np.load('train_targets.npy')
x_test = np.load('val_batches.npy')
y_test = np.load('val_targets.npy')

maxf = np.max(np.array([np.max(np.abs(x_train)),np.max(np.abs(x_test))]))

x_train = np.abs(x_train)/maxf
x_test = np.abs(x_test)/maxf

INPUT_SHAPE = x_train.shape[1:]

nsample = x_train.shape[0]

print('nsample = ',nsample)


ssplit = [128, 512, 1024, nsample/10, nsample/5, nsample/2, nsample]


nsplit = len(ssplit) 

#==============================================================================
# Train
#==============================================================================  
# 

reps = 32

pre_epochs = np.zeros((nsplit,reps))
folder = './LogsC/longitudinal/'
for k in range(reps):

	if not os.path.exists(folder+str(k)):
		os.makedirs(folder+str(k))
	
	random_emb = np.random.randn(NUM_CLASSES,NUM_OUTPUTS)
	random_emb = (random_emb-np.min(random_emb))
	random_emb = (random_emb/(np.max(np.abs(random_emb))))
	print(random_emb)

	i=0
	emb_train = np.ones((y_train.shape[0],NUM_OUTPUTS))
	for x in y_train[:]:
		idx = np.argmax(x)
		emb_train[i]=random_emb[idx]
		i=i+1

	i=0
	emb_test = np.ones((y_test.shape[0],NUM_OUTPUTS))
	for x in y_test[:]:
		idx = np.argmax(x)
		emb_test[i]=random_emb[idx]
		i=i+1

	for i in range(5,nsplit):
		a=k%(nsample/ssplit[i])
		div=4
		
		print(ssplit[i])
		print(a,nsample/ssplit[i])
			
		x_split = x_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		y_split = y_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		emb_split = emb_train[(ssplit[i]*a):(ssplit[i]*(a+1))]


		model1 = emb_cnn_pre(INPUT_SHAPE, NUM_OUTPUTS)
		model1.compile(optimizer='rmsprop', loss='mse')

		earlyStopping=EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto', restore_best_weights=True)
		history = model1.fit([x_split[k%div::div]], [emb_split[k%div::div]],
							batch_size=BATCH,
							epochs=EPOCHS,
							callbacks=[earlyStopping],
							shuffle=True,
							verbose=0)

		pre_epochs[i,k] = len(history.history['loss'])
		print('Pre-training stopped at epoch ',pre_epochs[i,k])
		
		batch_norm=True
		if (i<5):
			batch_norm=False

		model = emb_cnn_full(model1, INPUT_SHAPE, NUM_CLASSES, embodied=False,batch_norm=batch_norm)

		custom_callback = CustomCallback(k,i,x_test,y_test,folder)
		csv_logger = CSVLogger(folder+str(k)+'/log_'+"{:03d}".format(i)+'.csv')

		folder_chkpts = 'checkpoints/'+folder+str(k)+'/'+str(i)	
		if not os.path.exists(folder_chkpts):
			os.makedirs(folder_chkpts)
			
		filepath=folder_chkpts+"/model-{epoch:02d}"+"{:03d}".format(i)+".hdf5"
		checkpoints = ModelCheckpoint(filepath, verbose=0, save_best_only=False, mode='auto', period=1)
		callbacks = [custom_callback,csv_logger,checkpoints]

		model.compile(loss={"class_output": 'categorical_crossentropy'},
						optimizer='adam',
						metrics={"class_output": ['accuracy',acc_likelihood,acc_threshold]})

		history = model.fit([x_split], [y_split],
							batch_size=BATCH/2,
							epochs=EPOCHS*2,
							callbacks=callbacks,
							shuffle=True,
							verbose=0,
							validation_data=([x_test], [y_test]))
		K.clear_session()

np.savetxt(folder+'pre_epochs.cvs',pre_epochs,delimiter=',')
