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

from sklearn.model_selection import train_test_split


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


def create_ceps3d_all_data():
    genre_list = GENRE_LIST
    base_dir = GENRE_DIR
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps3d.npy")):
            ceps3d = np.load(fn)

            X.append(ceps3d)
            y.append(label)

    all_x_data = np.array(X)
    all_y_data = np.array(y)

    x_data_path = GENRE_DIR + 'x_3d_all_data'
    y_data_path = GENRE_DIR + 'y_3d_all_data'

    np.save(x_data_path, all_x_data)
    np.save(y_data_path, all_y_data)
    print("Written", x_data_path)
    print("Written", y_data_path)


def create_ceps_all_data():
    genre_list = GENRE_LIST
    base_dir = GENRE_DIR
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

    all_x_data = np.array(X)
    all_y_data = np.array(y)

    x_data_path = GENRE_DIR + 'x_all_data'
    y_data_path = GENRE_DIR + 'y_all_data'

    np.save(x_data_path, all_x_data)
    np.save(y_data_path, all_y_data)
    print("Written", x_data_path)
    print("Written", y_data_path)

