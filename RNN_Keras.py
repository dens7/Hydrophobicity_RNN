# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:37:22 2019

@author: tambi
"""

import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os
import random
import pickle
import time
import tensorflow
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import sys
import gc


def choose_validation_set_indices(file_name, index):
    """
    Chooses the test_set according to the process number. The
    :param file_name:
    :param index: The model is learnt on a different part of the data set every time
    :return:
    """
    pickle_in = open(file_name, 'rb')
    nfold_arr = pickle.load(pickle_in)  # For the purpose of this project, n=5
    pickle_in.close()
    # nfold_arr[index] stores another list which contains 20% of the total no. of indices of validation set (8 surfaces)
    validation_set = nfold_arr[index]
    return validation_set


def load_pickle_array(folder, file):
    """
    This loads or converts the pickled file into the training set and validation set data
    :param folder: the folder that contains the training/validaiton data
    :param file:
    :return:
    """
    pickle_in = open(folder + file, "rb")  # '+' performs the join operation
    arr = np.array(pickle.load(pickle_in))  # converts the list to a numpy array
    pickle_in.close()
    print(file, 'loaded')
    return arr


if __name__ == "__main__":

    index = int(sys.argv[1])
    file_name = sys.argv[2]
    nDeep = sys.argv[3]
    validate_input_folder = train_input_folder = sys.argv[4]
    output_folder = ""
    norm_method = sys.argv[5]  # referring to the choice of normalizing frame-wise or all-wise
    img_rows = int(sys.argv[6])  # the number of rows in the image
    img_cols = int(sys.argv[7])
    batch_size = 128  # Batch size
    nb_epoch = 50  # Number of times data is passed through network

    start_time = time.time()

    # Loading the train and test set
    X_train = load_pickle_array(train_input_folder, 'X_train_np.pickle').astype('float16')
    X_validate = load_pickle_array(validate_input_folder, 'X_validate_np.pickle').astype('float16')
    y_train = load_pickle_array(train_input_folder, 'y_train.pickle').astype('float32')
    y_validate = load_pickle_array(validate_input_folder, 'y_validate.pickle').astype('float32')

    # Normalize with max value of entire density array
    if (norm_method == 'all'):
        X_train /= np.max(X_train)
        X_validate /= np.max(X_validate)
        print('Arrays normalized with max value')

    # Normalize with max value of each frame
    if (norm_method == 'frame'):
        train_max = np.zeros((X_train.shape[0], 1))
        for frame in range(nDeep):
            next = np.array([np.max(x) for x in X_train[:, :, :, frame]])
            next = np.reshape(next, (next.shape[0], 1))
            train_max = np.hstack((train_max, next))
        train_max = np.delete(train_max, 0, axis=1)

        val_max = np.zeros((X_validate.shape[0], 1))
        for frame in range(nDeep):
            next = np.array([np.max(x) for x in X_validate[:, :, :, frame]])
            next = np.reshape(next, (next.shape[0], 1))
            val_max = np.hstack((val_max, next))
        val_max = np.delete(val_max, 0, axis=1)

        X_train /= train_max[:, np.newaxis, np.newaxis, :]  # np.new
        X_validate /= val_max[:, np.newaxis, np.newaxis, :]
        print('Arrays normalized with frame-wise max value')

    # To keep only validation set entries in validation array, and to delete these from the training set
    y_unique = np.unique(y_validate)
    validate_entries = y_unique[choose_validation_set_indices(file_name, index)]
    for val in validate_entries:
        # np.where returns the array of indices where the values in the training set coincide with the values in the
        # validation set. np.delete removes these values (from X and y)
        X_train = np.delete(X_train, np.where(y_train == val), axis=0)
        y_train = np.delete(y_train, np.where(y_train == val), axis=0)

    unflattened_shape = X_train.shape
    print("Shape of X_train:", unflattened_shape)
    X_train = X_train.reshape((unflattened_shape[0], -1, nDeep))

    # Definition of size of problem
    print('Defining network')
    nb_classes = len(np.unique(y_train))  # Number of classes

    # Standardization of the labels of training and validation set
    y_validate = (y_validate - np.mean(y_train)) / np.std(y_train)
    y_train = (y_train - np.mean(y_train)) / np.std(y_train)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_validate.shape)
    print('y_test shape:', y_validate)

    # Shape of input - 32x32x1
    in_shape = ( nDeep)

    model = Sequential()


    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Convolution2D(5, (5, 5), kernel_initializer='he_normal', activation = 'relu', name='conv2'))
    model.add(Convolution2D(12, (5, 5), kernel_initializer='he_normal', activation='relu', name='conv2'))
    # model.add(Convolution2D(25, (5, 5), kernel_initializer='he_normal', activation = 'relu', name='conv2'))
    # model.add(Convolution2D(25, (5, 5), kernel_initializer='he_normal', activation = 'relu', name='conv2'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(180, activation='relu', kernel_initializer='he_normal', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal', name='dense2'))
    model.add(Dropout(0.5))
    # model.add(Dense(nb_classes, activation='softmax', kernel_initializer='he_normal', name='dense3'))

    model.add(Dense(1, activation='linear', kernel_initializer='he_normal', name='dense3'))

    # adamax = keras.optimizers.Adamax(lr=0.000, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    # model.optimizer.lr.set_value(0)

    model.compile(loss='mse', optimizer='adamax', metrics=["mse"])
    print('Model compiled, now fitting')

    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                        validation_split=0.2)
    print('Model fit, now predicting for validation set')

    y_predict = model.predict(X_validate)

    v_i = np.mean(y_predict[y_validate == y_unique[i]]) * np.std(y_train) + np.mean(y_train)
    print('i - ', i, ',', y_unique[i], '-', v_i)

    data = {
        'y_train': np.unique(y_train),
        'y_validate': np.unique(y_validate),
        'i': i,
        'j': j,
        'k': k,
        'y_unique': y_unique,
        'v_i': v_i,
        'v_j': v_j,
        'v_k': v_k,
        'std': np.std(y_train_curr),
        'mean': np.mean(y_train_curr),
        'yPred': yPred,
        'validate_entries': validate_entries,
        'X_val': X_validate_curr,
        'y_val': y_validate_curr
    }
    file_name = output_folder + 'regress_store_' + str(i) + '_j_' + str(j) + '_k_' + str(k) + '.pickle'
    pickle_in = open(file_name, 'wb')
    pickle.dump(data, pickle_in)
    pickle_in.close()

    file_name = output_folder + 'model_store_' + str(i) + '_j_' + str(j) + '_k_' + str(k) + '.pickle'
    pickle_in = open(file_name, 'wb')
    pickle.dump(model, pickle_in)
    pickle_in.close()

    ## Plot validation and training loss ##
    # plt.plot(history.history['val_loss'])
    # plt.savefig("val_loss.png")
    # plt.plot(history.history['loss'])
    # plt.savefig("train_loss.png")

    gc.collect()

    end_time = (time.time() - start_time) / 60
    print('time of run - {:0.1f} mins'.format(end_time))
