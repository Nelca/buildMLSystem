
# coding: utf-8

# In[8]:

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
# GENRE_LIST.append("blues")
# GENRE_LIST.append("classical")
GENRE_LIST.append("country")
GENRE_LIST.append("disco")
GENRE_LIST.append("hiphop")
GENRE_LIST.append("jazz")
GENRE_LIST.append("metal")
GENRE_LIST.append("pop")
GENRE_LIST.append("reggae")
GENRE_LIST.append("rock")


# In[2]:

# checking data shapes
fn = "../data/songData/genres/blues/blues.00000.wav"
y, sr = librosa.load(fn)

mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=30)
y_harmonic, y_percussive = librosa.effects.hpss(y)
percus_mfcc = librosa.feature.mfcc(y=y_percussive, sr=sr,n_mfcc=30)
harmonic_mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr,n_mfcc=30)


# In[3]:

data = []
data.append(mfcc)
data.append(percus_mfcc)
data.append(harmonic_mfcc)
data = np.array(data)
print(data.shape)


# In[4]:

def create_data_for_conv(genre_list=GENRE_LIST):
    os.chdir(GENRE_DIR)
    for genre in genre_list:
        glob_wav = os.path.join(genre, "*.wav")
        for fn in glob.glob(glob_wav):
            create_mfcc_3d(fn)


# In[5]:

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


# In[6]:

import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'create_dataset_mfcc_3d.ipynb'])


# In[9]:

create_data_for_conv()

