# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

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


def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Written", data_fn)


def create_ceps(fn):
    y, sr = librosa.load(fn)
    ceps = librosa.feature.mfcc(y=y, sr=sr)

    write_ceps(ceps, fn)

def read_ceps3d_with_train_test(base_dir=GENRE_DIR, recreate_data=False):
    X_train_path = GENRE_DIR + 'X_3d_train'
    X_test_path =  GENRE_DIR + 'X_3d_test'
    y_train_path= GENRE_DIR + 'y_3d_train'
    y_test_path = GENRE_DIR + 'y_3d_test'
    if (recreate_data) :
        x_data_path = GENRE_DIR  + 'x_3d_all_data.npy'
        y_data_path = GENRE_DIR + 'y_3d_all_data.npy'

        all_x_data = np.load(x_data_path)
        all_y_data = np.load(y_data_path)

        X_train, X_test, y_train, y_test = train_test_split(
                all_x_data, all_y_data, test_size=0.4, random_state=13)
        print("check shape of y_train")
        print(y_train.shape)

        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        print("check shape of categoricaled y_train")
        print(y_train.shape)

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

def read_mfcc_with_train_test(base_dir=GENRE_DIR, recreate_data=False):
    X_train_path = GENRE_DIR + 'X_mfcc_train'
    X_test_path =  GENRE_DIR + 'X_mfcc_test'
    y_train_path= GENRE_DIR + 'y_mfcc_train'
    y_test_path = GENRE_DIR + 'y_mfcc_test'
    if (recreate_data) :
        x_data_path = GENRE_DIR  + 'x_mfcc_all_data.npy'
        y_data_path = GENRE_DIR + 'y_mfcc_all_data.npy'

        all_x_data = np.load(x_data_path)
        all_y_data = np.load(y_data_path)

        X_train, X_test, y_train, y_test = train_test_split(
                all_x_data, all_y_data, test_size=0.4, random_state=13)
        print("check shape of y_train")
        print(y_train.shape)

        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        print("check shape of categoricaled y_train")
        print(y_train.shape)

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

def read_ceps_with_train_test(base_dir=GENRE_DIR, recreate_data=False):
    X_train_path = GENRE_DIR + 'X_train'
    X_test_path =  GENRE_DIR + 'X_test'
    y_train_path= GENRE_DIR + 'y_train'
    y_test_path = GENRE_DIR + 'y_test'
    if (recreate_data) :
        x_data_path = GENRE_DIR  + 'x_all_data.npy'
        y_data_path = GENRE_DIR + 'y_all_data.npy'

        all_x_data = np.load(x_data_path)
        all_y_data = np.load(y_data_path)

        X_train, X_test, y_train, y_test = train_test_split(
                all_x_data, all_y_data, test_size=0.4, random_state=13)

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

def read_ceps(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            t_ceps = ceps.transpose()
            t_num_ceps = len(t_ceps)
            t_ceps_mean = np.mean(t_ceps[int(t_num_ceps / 10):int(t_num_ceps * 9 / 10)], axis=0)

            X.append(t_ceps_mean)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    os.chdir(GENRE_DIR)
    glob_wav = os.path.join(sys.argv[1], "*.wav")
    print(glob_wav)
    for fn in glob.glob(glob_wav):
        create_ceps(fn)
