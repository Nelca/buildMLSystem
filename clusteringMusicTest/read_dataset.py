import os
import glob
import sys

import numpy as np
import librosa

from sklearn.model_selection import train_test_split
import keras


GENRE_DIR = "../data/songData/genres/"
GENRE_LIST = []
GENRE_LIST.append("blues")
GENRE_LIST.append("classical")
GENRE_LIST.append("country")
GENRE_LIST.append("disco")
GENRE_LIST.append("hiphop")
GENRE_LIST.append("jazz")
GENRE_LIST.append("metal")
GENRE_LIST.append("pop")
GENRE_LIST.append("reggae")
GENRE_LIST.append("rock")


def read_dataset_with_train_test(data_type="", base_dir=GENRE_DIR, recreate_data=False):
    X_train_path = GENRE_DIR + 'X_' + data_type + '_train'
    X_test_path =  GENRE_DIR + 'X_' + data_type + '_test'
    y_train_path= GENRE_DIR + 'y_' + data_type + '_train'
    y_test_path = GENRE_DIR + 'y_' + data_type + '_test'
    if (recreate_data) :
        x_data_path = GENRE_DIR  + 'x_' + data_type + '_all_data.npy'
        y_data_path = GENRE_DIR + 'y_' + data_type + '_all_data.npy'

        all_x_data = np.load(x_data_path)
        all_y_data = np.load(y_data_path)

        X_train, X_test, y_train, y_test = train_test_split(
                all_x_data, all_y_data, test_size=0.4, random_state=13)

        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        print("recreated data as follow.")
        print("X_train shape is " + X_train.shape)
        print("y_train shape is " + y_train.shape)
        print("X_test shape is " + X_test.shape)
        print("y_test shape is " + y_test.shape)

        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        np.save(y_train_path, y_train)
        np.save(y_test_path, y_test)

    else :
        X_train_path = X_train_path + '.npy'
        X_test_path = X_test_path + '.npy'
        y_train_path = y_train_path + '.npy'
        y_test_path = y_test_path + '.npy'

        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)

    return X_train, X_test, y_train, y_test


def read_ceps3d_with_train_test(base_dir=GENRE_DIR, recreate_data=False):

    X_train, X_test, y_train, y_test = read_dataset_with_train_test(data_type="3d", base_dir=base_dir, recreate_data=recreate_data)

    return X_train, X_test, y_train, y_test

def read_mfcc_with_train_test(base_dir=GENRE_DIR, recreate_data=False):

    X_train, X_test, y_train, y_test = read_dataset_with_train_test(data_type="mfcc", base_dir=base_dir, recreate_data=recreate_data)

    return X_train, X_test, y_train, y_test

def read_ceps_with_train_test(base_dir=GENRE_DIR, recreate_data=False):
    X_train, X_test, y_train, y_test = read_dataset_with_train_test(data_type="ceps", base_dir=base_dir, recreate_data=recreate_data)

    return X_train, X_test, y_train, y_test


