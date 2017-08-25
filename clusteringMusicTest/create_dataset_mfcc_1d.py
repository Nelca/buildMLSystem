
# coding: utf-8

# In[1]:

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

GENRE_DIR = "/home/minato/deep_learning/buildMLSystem/data/data/songData/genres/"
GENRE_LIST = []
#0
GENRE_LIST.append("blues")
#1
GENRE_LIST.append("classical")
#2
GENRE_LIST.append("country")
#3
GENRE_LIST.append("disco")
#4
GENRE_LIST.append("hiphop")
#5
GENRE_LIST.append("jazz")
#6
GENRE_LIST.append("metal")
#7
GENRE_LIST.append("pop")
#8
GENRE_LIST.append("reggae")
#9
GENRE_LIST.append("rock")


# In[2]:

def create_data_for_conv(genre_list=GENRE_LIST):
    os.chdir(GENRE_DIR)
    for genre in genre_list:
        glob_wav = os.path.join(genre, "*.wav")
        for fn in glob.glob(glob_wav):
            create_mfcc_1d(fn) 


# In[8]:

def create_mfcc_1d(fn):
    y, sr = librosa.load(fn)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=30)
    last_dim = mfcc.shape[1]
    if last_dim < 1293:
        add_dim = 1293 - last_dim
        add_list = np.zeros((30, add_dim))
        mfcc = np.append(mfcc, add_list, axis=1)
    elif  last_dim > 1293:
        mfcc = mfcc[:,:1293]
    
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".mfcc1d"
    np.save(data_fn, mfcc)
    print("Written", data_fn)


# In[4]:

def create_ceps1d_all_data():
    genre_list = GENRE_LIST
    base_dir = GENRE_DIR
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.mfcc1d.npy")):
            mfcc1d = np.load(fn)
            if mfcc1d[0][0].shape[0] != 1293:
                print("file shape is", mfcc1d.shape)
                print("fn is", fn)
            X.append(mfcc1d)
            y.append(label)

    print("loaded all data")
    all_x_data = np.array(X)
    all_y_data = np.array(y)

    x_data_path = GENRE_DIR + 'x_1d_all_data'
    y_data_path = GENRE_DIR + 'y_1d_all_data'

    np.save(x_data_path, all_x_data)
    np.save(y_data_path, all_y_data)
    print("Written", x_data_path)
    print("Written", y_data_path)


# In[5]:

def prepare_mfcc3d_data():
    genre_list = GENRE_LIST
    base_dir = GENRE_DIR
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.mfcc3d.npy")):
            mfcc3d = np.load(fn)
            last_dim = mfcc3d[0][0].shape[0]
            if last_dim < 1293:
                add_dim = 1293 - last_dim
                add_list = np.zeros((3, 30, add_dim))
                mfcc3d_formated_data = np.append(mfcc3d, add_list, axis=2)
                np.save(fn, mfcc3d_formated_data)
                print("recreate data of", fn)


# In[6]:

import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'create_dataset_mfcc_1d.ipynb'])


# In[10]:

create_data_for_conv()


# In[ ]:

create_ceps1d_all_data()

