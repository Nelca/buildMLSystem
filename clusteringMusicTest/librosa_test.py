import librosa

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

