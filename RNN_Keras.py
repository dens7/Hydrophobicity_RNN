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
    Chooses the validation_set according to the process number. The
    :param file_name:
    :param index: The model is learnt on a different part of the data set every time
    :return:
    """
    pickle_in = open(file_name, 'rb')
    nfold_arr = pickle.load(pickle_in)  # For the purpose of this project, n=5
    pickle_in.close()
    # nfold_arr[index] stores another list
    validation_set = nfold_arr[index]
    return validation_set


def load_pickle_array(folder, file):
    pickle_in = open(folder + file, "rb")
    arr = pickle.load(pickle_in)
    pickle_in.close()
    arr = np.array(arr)
    print(file, 'loaded')
    return arr


if __name__ == "__main__":

    index = int(sys.argv[1])
    file_name = sys.argv[2]
    nDeep = sys.argv[3]
    validate_input_folder = train_input_folder = sys.argv[4]
    output_folder = ""
    i, j, k = index, index, index
    norm_method = sys.argv[5]
    img_rows = int(sys.argv[6])
    img_cols = int(sys.argv[7])
    # file_name = int(sys.argv[8])
    batch_size = 128  # Batch size
    nb_epoch = 50  # Number of times data is passed through network
    n_split = -1  # Ensures reading of arrays as 1 contiguous array file rather than n_split different pickles

    start_time = time.time()


    print(train_input_folder)

    if (n_split == -1):
        X_train_curr = load_pickle_array(train_input_folder, 'X_train_np.pickle')
        X_validate_curr = load_pickle_array(validate_input_folder, 'X_validate_np.pickle')
        y_train_curr = load_pickle_array(train_input_folder, 'y_train.pickle')
        y_validate_curr = load_pickle_array(validate_input_folder, 'y_validate.pickle')
    elif (n_split > 0):
        for i in range(n_split):
            train_x_str = 'X_train_np_' + str(i)
            val_x_str = 'X_validate_np_' + str(i)
            train_y_str = 'y_train_' + str(i)
            val_y_str = 'y_validate_' + str(i)
            if (i == 0):
                X_train_curr = load_pickle_array(train_input_folder, train_x_str)
                X_validate_curr = load_pickle_array(validate_input_folder, val_x_str)
                y_train_curr = load_pickle_array(train_input_folder, train_y_str)
                y_validate_curr = load_pickle_array(validate_input_folder, val_y_str)
            else:
                X_train_curr_temp = load_pickle_array(train_input_folder, train_x_str)
                X_validate_curr_temp = load_pickle_array(validate_input_folder, val_x_str)
                y_train_curr_temp = load_pickle_array(train_input_folder, train_y_str)
                y_validate_curr_temp = load_pickle_array(validate_input_folder, val_y_str)
                X_train_curr = np.concatenate((X_train_curr, X_train_curr_temp))
                X_validate_curr = np.concatenate((X_validate_curr, X_validate_curr_temp))
                y_train_curr = np.concatenate((y_train_curr, y_train_curr_temp))
                y_validate_curr = np.concatenate((y_validate_curr, y_validate_curr_temp))
    X_train_curr = X_train_curr.astype('float16')
    X_validate_curr = X_validate_curr.astype('float16')
    print('Converted X arrays to float16')

    ## Converting y arrays to float16 does not lead to substantial size reduction, and kills all values ##
    y_train_curr = y_train_curr.astype('float32')
    y_validate_curr = y_validate_curr.astype('float32')

    # ~~~~~~~~~~ Normalize with max value of entire density array ~~~~~~~~~~ #
    if (norm_method == 'all'):
        X_train_curr /= np.max(X_train_curr)
        X_validate_curr /= np.max(X_validate_curr)
        print('Arrays normalized with max value')

    ## ~~~~~~~~~~ Normalize with max value of each frame ~~~~~~~~~~ #
    if (norm_method == 'frame'):
        train_max_1 = np.zeros((X_train_curr.shape[0], 1))
        for count_1 in range(nDeep):
            train_max_0 = np.array([np.max(x) for x in X_train_curr[:, :, :, count_1]])
            train_max_0 = np.reshape(train_max_0, (train_max_0.shape[0], 1))
            train_max_1 = np.hstack((train_max_1, train_max_0))
        train_max_1 = np.delete(train_max_1, 0, axis=1)

        val_max_1 = np.zeros((X_validate_curr.shape[0], 1))
        for count_2 in range(nDeep):
            val_max_0 = np.array([np.max(x) for x in X_validate_curr[:, :, :, count_2]])
            val_max_0 = np.reshape(val_max_0, (val_max_0.shape[0], 1))
            val_max_1 = np.hstack((val_max_1, val_max_0))
        val_max_1 = np.delete(val_max_1, 0, axis=1)

        X_train_curr /= train_max_1[:, np.newaxis, np.newaxis, :] # np.new
        X_validate_curr /= val_max_1[:, np.newaxis, np.newaxis, :]
        print('Arrays normalized with frame-wise max value')

    y_unique = np.unique(y_validate_curr)
    validate_entries = y_unique[choose_validation_set_indices(file_name, index)]
    for val in validate_entries:
        X_train_curr = np.delete(X_train_curr, np.where(y_train_curr == val), axis=0)
        y_train_curr = np.delete(y_train_curr, np.where(y_train_curr == val), axis=0)
    # Keep only validation set entries in validation array
    shape = np.shape(X_train_curr)

    if (nDeep == 1):
        X_train_curr = X_train_curr[:, :, :, 0]

    X_train_curr = X_train_curr.reshape((shape[0], shape[1], shape[2], nDeep))

    # Definition of size of problem
    print('Defining network')
    nb_classes = len(np.unique(y_train_curr))  # Number of classes

    y_train_1 = (y_train_curr - np.mean(y_train_curr)) / np.std(y_train_curr)
    y_validate_1 = (y_validate_curr - np.mean(y_train_curr)) / np.std(y_train_curr)
    y_train_1 = (y_train_curr - np.mean(y_train_curr)) / np.std(y_train_curr)

    print('X_train shape:', X_train_curr.shape)
    print('Y_train shape:', len(y_train_curr))
    print('X_test shape:', X_validate_curr.shape)

    # Shape of input - 32x32x1
    in_shape = (img_rows, img_cols, nDeep)

    model = Sequential()

    # model.add(Convolution2D(2, (5, 5), activation = 'relu', kernel_initializer='he_normal', input_shape=in_shape, name='conv1'))

    # model.add(Convolution2D(12, (5, 5), activation = 'relu', kernel_initializer='he_normal', input_shape=in_shape, name='conv1'))

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

    history = model.fit(X_train_curr, y_train_1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                        validation_split=0.2)
    print('Model fit, now predicting for validation set')

    yPred = model.predict(X_validate_curr)

    # acc = np.sum(yPred==y_validate)/len(y_validate)*100

    v_i = np.mean(yPred[y_validate_curr == y_unique[i]]) * np.std(y_train_curr) + np.mean(y_train_curr)
    v_j = np.mean(yPred[y_validate_curr == y_unique[j]]) * np.std(y_train_curr) + np.mean(y_train_curr)
    v_k = np.mean(yPred[y_validate_curr == y_unique[k]]) * np.std(y_train_curr) + np.mean(y_train_curr)

    print('i - ', i, ',', y_unique[i], '-', v_i)
    print('j - ', j, ',', y_unique[j], '-', v_j)
    print('k - ', k, ',', y_unique[k], '-', v_k)

    data = {
        'y_train': np.unique(y_train_curr),
        'y_validate': np.unique(y_validate_curr),
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
    # plt.clf()

    gc.collect()

    end_time = (time.time() - start_time) / 60
    print('time of run - {:0.1f} mins'.format(end_time))
