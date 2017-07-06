
# coding: utf-8

# In[20]:

import os
import glob
import sys
import numpy as np

from sklearn.model_selection import train_test_split
import keras
import librosa

import librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
get_ipython().magic('matplotlib inline')

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


# In[2]:

def create_data_for_conv(genre_list=GENRE_LIST):
    os.chdir(GENRE_DIR)
    for genre in genre_list:
        glob_wav = os.path.join(genre, "*.wav")
        for fn in glob.glob(glob_wav):
            create_mfcc_3d(fn)


# In[3]:

def create_mfcc_3d(fn):
    y, sr = librosa.load(fn)
    y_harmonic, y_pertcussive = librosa.effects.hpss(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=30)
    harmonic_mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr,n_mfcc=30)
    percus_mfcc = librosa.feature.mfcc(y=y_pertcussive, sr=sr,n_mfcc=30)
    data = []
    data.append(mfcc)
    data.append(percus_mfcc)
    data.append(harmonic_mfcc)
    data = np.array(data)
    
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".mfcc3d"
    np.save(data_fn, data)
    print("Written", data_fn)


# In[23]:

X = []

fn = "../data/songData/genres/blues/blues.00000.mfcc3d.npy"
mfcc3d = np.load(fn)
X.append(mfcc3d)
print(mfcc3d.shape)

fn = "../data/songData/genres/blues/blues.00000.mfcc3d.npy"
mfcc3d = np.load(fn)
X.append(mfcc3d)

all_x_data = np.array(X)
print(all_x_data.shape)


# In[24]:

def create_ceps3d_all_data():
    genre_list = GENRE_LIST
    base_dir = GENRE_DIR
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.mfcc3d.npy")):
            mfcc3d = np.load(fn)
            X.append(mfcc3d)
            y.append(label)

    print("loaded all data")
    all_x_data = np.array(X)
    all_y_data = np.array(y)

    x_data_path = GENRE_DIR + 'x_3d_all_data'
    y_data_path = GENRE_DIR + 'y_3d_all_data'

    np.save(x_data_path, all_x_data)
    np.save(y_data_path, all_y_data)
    print("Written", x_data_path)
    print("Written", y_data_path)


# In[5]:

import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'create_dataset_mfcc_3d.ipynb'])


# In[25]:



