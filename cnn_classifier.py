#!/usr/bin/env python
# coding: utf-8


import keras
import numpy as np
import pandas
import os
from Bio import SeqIO
import argparse
import pickle

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv2D, Conv1D
from keras.layers import AveragePooling1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



#parse input argument(s)
parser = argparse.ArgumentParser()
parser.add_argument('factor_number', type=int)

args = parser.parse_args()
factor_number = args.factor_number


# ## Import data

def parse_fastq(file_name):
    input_seq_iterator = SeqIO.parse(file_name, "fastq")
    return [str(record.seq) for record in input_seq_iterator]


def sequences_to_numpy(seq_list):
    m = len(seq_list)
    seq_arr = np.array([list(seq) for seq in seq_list])
    d = len(seq_list[0])
    X = np.zeros((m, d, 5))
    
    for i, nuc in enumerate('ACGT', start=1):
        seq_idx, position_idx = np.where(seq_arr == nuc)
        X[seq_idx, position_idx, i] = 1
    return X


    
#parse the dataset to find the corresponding files for each factor
selex_files = np.loadtxt('selex_files.txt', dtype=str) #name of all files

#first part of file_name is the protein name
factors = np.unique([s.split('_')[0] for s in selex_files]) 

#select the factor based on the input
factor = factors[factor_number]

#files that correspond to this factor
factor_files = selex_files[np.array([factor in s for s in selex_files])]
factor_barcodes = np.unique([s.split('_')[1] for s in factor_files])
files_by_bc = []
for bc in factor_barcodes:
    files = factor_files[np.array([bc in s for s in factor_files])]
    corresponding_bg_file = selex_files[np.array([(bc in s)&('ZeroCycle' in s) for s in selex_files])][0]
    files_with_bg = np.append(files,corresponding_bg_file)
    files_with_bg_sorted_by_cycle = files_with_bg[np.argsort([int(f.split('_')[3]) for f in files_with_bg])]
    files_by_bc.append(files_with_bg_sorted_by_cycle)


#select cycles
avg_auc_file = os.path.join('../RBP_motif_cluster/scripts/param/selex/', f'cl3{factor}_avg_auc.txt')
if os.path.isfile(avg_auc_file):
    avg_auc = np.loadtxt(avg_auc_file)
    bg_round = np.argmax(avg_auc)
    pos_round = bg_round+1
else:
    print('pre-selection failed')
    bg_round = len(files_by_bc[0])-2
    pos_round = len(files_by_bc[0])-1
    
print('bg round chosen:' + str(files_by_bc[0][bg_round].split('_')[3]))
print('ps round chosen:' + str(files_by_bc[0][pos_round].split('_')[3]))

file_name = f'{factor}_' + str(files_by_bc[0][bg_round].split('_')[3]) + 'vs' + str(files_by_bc[0][pos_round].split('_')[3])

bg_file  = os.path.join('../rbp_scratch/data', files_by_bc[0][bg_round])
pos_file = os.path.join('../rbp_scratch/data', files_by_bc[0][pos_round])

bg = parse_fastq(bg_file)
ps = parse_fastq(pos_file)


y = np.concatenate([np.ones(len(ps)), np.zeros(len(bg))])
X = np.concatenate([sequences_to_numpy(ps), sequences_to_numpy(bg)])


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Convolutional NN
def seq_CNN(input_shape):
    """
    Implementation of the CNN.
    
    Arguments:
    input_shape -- shape of single sequences
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(64, 5, strides = 1, padding='same', name='conv0')(X_input)
    X = BatchNormalization(axis = 2, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling1D(2, name='max_pool')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(128, 7, strides = 1, padding='valid', name='conv1')(X_input)
    X = BatchNormalization(axis = 2, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # MAXPOOL
    X = MaxPooling1D(2, name='max_pool2')(X)
    

    # FLATTEN X (means convert it to a vector) -> FULLYCONNECTED -> FULLYCONNECTED
    X = Flatten()(X)
    X = layers.Dropout(0.2)(X)
    X = Dense(25, name='fc')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = layers.Dropout(0.2)(X)
    X = Dense(10, name='fc2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    
    X = layers.Dropout(0.2)(X)
    X = Dense(1, activation='sigmoid', name='output')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='seq_cnn')

    
    ### END CODE HERE ###
    
    return model

#create model
seq_cnn = seq_CNN(X_train.shape[1:])

#compile the model
seq_cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=[tf.keras.metrics.AUC(name='auc')])

seq_cnn.fit(x = X_train, y = y_train, epochs = 30, batch_size = 64, validation_data=(X_test, y_test))


with open(f'cnn/{file_name}.pickle', 'wb') as file:
    pickle.dump(seq_cnn.history.history, file)



#plot model's performance and auc over epochs
y_pred_keras = seq_cnn.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(7,3))
plt.subplots_adjust(wspace=0.4)

ax1.plot([0, 1], [0, 1], 'k--')
ax1.plot(fpr_keras, tpr_keras, label='CNN (area = {:.3f})'.format(auc_keras))
ax1.set_xlabel('False positive rate')
ax1.set_ylabel('True positive rate')
ax1.set_title('ROC curve')
ax1.legend(loc='best')


val_auc = seq_cnn.history.history['val_auc']
train_auc = seq_cnn.history.history['auc']
xpos = np.arange(1,len(val_auc)+1)

ax2.plot(xpos, val_auc, label='validation AUC')
ax2.plot(xpos, train_auc, label='training AUC')
ax2.set_xlabel('epochs')
ax2.set_ylabel('AUC')
ax2.legend(loc='best')

fig.suptitle(file_name)
plt.savefig('cnn/'+ file_name +'.pdf', bbox_inches='tight')
