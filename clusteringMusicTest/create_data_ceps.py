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

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


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

def createCepsSS():
    file_path = "../data/songData/genres/x_ceps_all_data.npy"
    ss_file_path = "./savedStanderdScaler/ceps_ss.pkl"
    print("load all ceps data")
    all_x_data = np.load(file_path)
    print("loaded!!")
    print("creating standard scaler model...")
    n = all_x_data.shape[0]
    reshaped_data = all_x_data.reshape(n, -1)

    ss = StandardScaler()
    ss.fit(reshaped_data)
    joblib.dump(ss, ss_file_path)
    print("saved ceps standard scaler.")
    print("file is " + ss_file_path)

if __name__ == "__main__":
    os.chdir(GENRE_DIR)
    glob_wav = os.path.join(sys.argv[1], "*.wav")
    print(glob_wav)
    for fn in glob.glob(glob_wav):
        create_ceps(fn)
