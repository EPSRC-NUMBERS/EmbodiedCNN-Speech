''' 
functions to build models for experiments on spoken digits recognition are here
'''

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, concatenate
from keras.regularizers import l1,l2
from customs import TerminateOnBaseline

l1_lambda = 0.0001
drop_0 = 0.25
drop_1 = 0.5
hidden = 128
						  
def emb_cnn_pre(features_shape, num_outputs, act='relu', act2='sigmoid'):
	#  merges convolutional blocks with embodied output for pre-training

	x = Input(name='inputs', shape=features_shape, dtype='float32')
	o = x
	
	o = base_conv(o, features_shape, act)

	o = Dense(num_outputs, activation=act2, kernel_initializer='glorot_uniform', name="emb_inout")(o) # embodied output
	# Create model and print network summary
	# Model(inputs=x, outputs=o).summary()
	
	return Model(inputs=x, outputs=o)

def emb_cnn_full(model1, features_shape, num_classes, act='relu', embodied=True, batch_norm=False):
	# Build the Full model

	emb_o = model1.get_layer('emb_inout').output
	o = model1.get_layer('flatten').output

	o = Dense(hidden, activation=act, kernel_initializer='glorot_uniform', name='dense')(o)
	if (batch_norm):
		o = BatchNormalization(name='dense_norm')(o) # this can increase accuracy with larger training sets

	if (embodied):
		o = concatenate([o, emb_o],axis=1,name="concatenate")

	# Predictions
	o = Dropout(0.5, name='dropout3')(o)
	o = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform', name='class_output')(o)

	# Print network summary
	if (embodied):
		model = Model(inputs=model1.input, outputs=[o,emb_o])
	else:
		model = Model(inputs=model1.input, outputs=o)
		
	model.summary()
	
	return model

def deep_cnn(features_shape, num_classes, act='relu',batch_norm=True):
	#baseline model

	x = Input(name='inputs', shape=features_shape, dtype='float32')
	o = x
	
	o = base_conv(o, features_shape, act)
	
	# Dense layer
	o = Dense(hidden, activation=act, kernel_initializer='glorot_uniform', name='lin')(o)
	if (batch_norm):
		o = BatchNormalization(name='dense_norm')(o) # this can increase accuracy with larger training sets
	o = Dropout(drop_1, name='dropout3')(o)
	
	# Predictions
	o = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform', name='class_output')(o)

	# Print network summary
	Model(inputs=x, outputs=o).summary()
	
	return Model(inputs=x, outputs=o)

def base_conv(i, features_shape, act='relu'):
	## CONVOLUTIONAL BLOCKS
	# Block 1
	o = Conv2D(64, (3, 3), activation=act, padding='same', strides=1, kernel_initializer='he_uniform', name='block1_conv', input_shape=features_shape)(i)
	o = MaxPooling2D((3, 3), strides=(3,2), padding='same', name='block1_pool')(o)
	o = BatchNormalization(name='block1_norm')(o)
	o = Dropout(drop_0, name='dropout0')(o)
	
	# Block 2
	o = Conv2D(64, (3, 3), activation=act, padding='same', strides=1, kernel_initializer='he_uniform', name='block2_conv')(o)
	o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
	o = BatchNormalization(name='block2_norm')(o)
	o = Dropout(drop_0, name='dropout1')(o)

	# Block 3
	o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, kernel_initializer='he_uniform', name='block3_conv')(o)
	o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block3_pool')(o)
	o = BatchNormalization(name='block3_norm')(o)
	o = Dropout(drop_0, name='dropout2')(o)

	# Flatten
	o = Flatten(name='flatten')(o)
	
	return (o)