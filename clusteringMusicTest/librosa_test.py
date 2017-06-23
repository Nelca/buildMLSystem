import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.cluster

import librosa
import librosa.display

file_path = "/home/minato/deep_learning/buildMLSystem/data/songData/genres/blues/blues.00018.wav"

y, sr = librosa.load(file_path)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print("tempo is ")
print(tempo)
print("")
print("beat frame is ")
print(beat_frames)
print("")
print("")

ceps = librosa.feature.mfcc(y=y, sr=sr)
print("ceps is ")
print(ceps)
print("")
print("ceps shape is ")
print(ceps.shape)
print("")

mfcc_delta = librosa.feature.delta(ceps)
print("mfcc delta is")
print(mfcc_delta)
print("")
print("mfcc delta is")
print(mfcc_delta.shape)
print("")
print("")

y_harmonic, y_percussive = librosa.effects.hpss(y)
print("harmonic is")
print(y_harmonic)
print("")
print("percussive is")
print(y_percussive)

#### ----- to test the laplas...


BINS_PER_OCTAVE = 12 * 3
N_OCTAVE = 7

C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                            bins_per_octave=BINS_PER_OCTAVE,\n",
    "                                        n_bins=N_OCTAVE * BINS_PER_OCTAVE),\n",

p
