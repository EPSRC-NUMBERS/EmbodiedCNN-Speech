import numpy as np
import keras.utils
train = np.load('train_batches.npy')
targets = np.load('train_targets.npy')

targets = np.max(range(1,10)*targets,axis=1)

ones = train[targets == 1]
twos = train[targets == 2]
threes = train[targets == 3]
fours = train[targets == 4]
fives = train[targets == 5]
sixes = train[targets == 6]
sevens = train[targets == 7]
eights = train[targets == 8]
nines = train[targets == 9]

twos = twos[:ones.shape[0]/2]
threes = threes[:ones.shape[0]/3]
fours = fours[:ones.shape[0]/4]
fives = fives[:ones.shape[0]/5]
sixes = sixes[:ones.shape[0]/6]
sevens = sevens[:ones.shape[0]/7]
eights = eights[:ones.shape[0]/8]
nines = nines[:ones.shape[0]/9]

zip_train = np.concatenate((ones,twos,threes,fours,fives,sixes,sevens,eights,nines))

zip_targets = np.ones((ones.shape[0],1))
for i in range(1,9):
    t_ = np.ones((ones.shape[0]/(i+1),1))*(i+1)
    zip_targets = np.concatenate((zip_targets,t_))

zip_targets = keras.utils.to_categorical(zip_targets-1)
print(zip_train.shape)
print(zip_targets.shape)
np.save('zip_train.npy',zip_train)
np.save('zip_targets.npy',zip_targets)

pre_items = nines.shape[0]

zip_pretrain = np.concatenate((ones[:pre_items],twos[:pre_items],threes[:pre_items],fours[:pre_items],fives[:pre_items],sixes[:pre_items]
    ,sevens[:pre_items],eights[:pre_items],nines[:pre_items]))


zip_pretargets = np.ones((pre_items,1))
for i in range(1,9):
    t_ = np.ones((pre_items,1))*(i+1)
    zip_pretargets = np.concatenate((zip_pretargets,t_))

zip_pretargets = keras.utils.to_categorical(zip_pretargets-1)

idxs = np.zeros(pre_items*9)
for i in range(pre_items):
    idxs[(i*9):((i+1)*9)] = (np.arange(9)*96)+i

print(idxs)

zip_pretrain = zip_pretrain[idxs.astype('int')]
zip_pretargets = zip_pretargets[idxs.astype('int')]

np.save('zip_pretrain.npy',zip_pretrain)
np.save('zip_pretargets.npy',zip_pretargets)

print(zip_pretrain.shape)
print(zip_pretargets.shape)