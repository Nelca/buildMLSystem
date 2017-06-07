# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sys
import os
import glob

import numpy as np
import scipy
import scipy.io.wavfile
import pdb

GENRE_DIR = "../data/songData/genres"
CHART_DIR = os.path.join("..", "charts")


def write_fft(fft_features, fn):
    """
    Write the FFT features to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"

    pdb.set_trace()

    np.save(data_fn, fft_features)
    print("Written", data_fn)


def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)

    pdb.set_trace()

    fft_features = abs(scipy.fft(X)[:1000])
    write_fft(fft_features, fn)


def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        assert(file_list), genre_dir
        for fn in file_list:
            fft_features = np.load(fn)

            X.append(fft_features[:2000])
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    #fn_list = glob.glob(os.path.join(sys.argv[1], "*.wav"))
    fn_list = glob.glob(sys.argv[1] +  "*.wav")
    pdb.set_trace()
    for fn in fn_list:
        create_fft(fn)

