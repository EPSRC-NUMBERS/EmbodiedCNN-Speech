'''
Auxiliary functions
'''

from keras.callbacks import Callback
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import spatial
import os, sys

def acc_likelihood(y_true, y_pred):
	return K.mean(K.max(y_pred*y_true,1))

def acc_threshold(y_true, y_pred): #counts recognition with likehood > 0.5
	y_threshold = K.cast(K.greater_equal(y_pred,0.5),dtype='float32')
	return K.mean(K.max(y_threshold*y_true,1))

def acc_top2(y_true, y_pred): #counts if correct is in the top 2 by likehood
	return top_k_categorical_accuracy(y_true,y_pred,k=2)

def ConfusionMatrix(y_pred,y_true,file_name='./cnf_max.cvs'): #calcualates the confusion matrix
		max_pred = np.argmax(y_pred, axis=1)
		max_true = np.argmax(y_true, axis=1)
		cnf_mat = confusion_matrix(max_true, max_pred)
		np.savetxt(file_name, cnf_mat, delimiter=",", header="1,2,3,4,5,6,7,8,9")
		return(cnf_mat)

def LikelihoodMatrix(y_pred,y_true,file_name='./like_mat.cvs'): #calculates the likehood matrix
		like_mat = np.zeros((9,9))
		max_true = np.argmax(y_true, axis=1)
		q=0
		for r in max_true:
			like_mat[r,:] += y_pred[q,:]
			q+=1
		np.savetxt(file_name, like_mat, delimiter=",", header="1,2,3,4,5,6,7,8,9")
		return(like_mat)

class CustomCallback(Callback): #prints results at the end of every epoch
	def __init__(self,rep,split,testx,testy,path_,embod=False):
		print('Starting Callback')
		self.rep = rep
		self.split = split
		self.x_test = testx
		self.y_test = testy
		self.folder = path_
		self.embod = embod

	def on_epoch_end(self, epoch, logs={}):
		print('Rep ', self.rep, ' - Split ', self.split)
		print('Current epoch : ', epoch)
		print('Training loss', logs.get('loss'))
		print('Test classification accuracy:', logs.get('val_acc'),logs.get('val_class_output_acc'))
		print('Test likelihood:',logs.get('val_acc_likelihood'),logs.get('val_class_output_acc_likelihood'))
		print('Test threshold accuracy:',logs.get('val_acc_threshold'),logs.get('val_class_output_acc_threshold'))
		if (epoch == 0 or epoch == 5 or epoch == 6 or epoch == 9 or epoch == 14 or epoch == 24 or epoch ==29 or epoch == 39 or epoch == 49):
			y = self.model.predict(self.x_test)
			if (self.embod):
				y = y[0]
			pathfile = './Results/conf_max/'+self.folder+"{:02d}".format(epoch)+'/'+"{:02d}".format(self.split)
			filename=pathfile+'/conf_mat_'+"{:03d}".format(self.rep)+'.csv'
			if not os.path.exists(pathfile):
				os.makedirs(pathfile)
			print(ConfusionMatrix(y,self.y_test,filename))
			pathfile = './Results/like_max/'+self.folder+"{:02d}".format(epoch)+'/'+"{:02d}".format(self.split)
			filename=pathfile+'/like_mat_'+"{:03d}".format(self.rep)+'.csv'
			if not os.path.exists(pathfile):
				os.makedirs(pathfile)
			LikelihoodMatrix(y,self.y_test,filename)

class TerminateOnBaseline(Callback):
	"""Callback that terminates training when the monitored variable reaches a specified baseline
	"""
	def __init__(self, monitor='acc', baseline=1, patience=1):
		super(TerminateOnBaseline, self).__init__()
		self.monitor = monitor
		self.baseline = baseline
		self.patience = patience
		self.counter = 0

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		acc = logs.get(self.monitor)
		if acc is not None:
			if acc >= self.baseline:
				self.counter+=1
				if self.counter >= self.patience:
					print('Epoch %d: Reached baseline, terminating training' % (epoch))
					self.model.stop_training = True