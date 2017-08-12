from sklearn.externals import joblib
from keras.models import load_model

def loadMfccStanderdScaler():
    saved_ss = joblib.load('./savedStanderdScaler/mfcc_ss.pkl')
    return saved_ss

def loadMfcc3dStanderdScaler():
    saved_ss = joblib.load('./savedStanderdScaler/mfcc_3d_ss.pkl')
    return saved_ss


def loadKerasModel(file_name):
    loaded_model = load_model('./savedModels/' + file_name + '.h5')
    return loaded_model


def loadCepsDenseModel():
    loaded_model = loadKerasModel('ceps_dense_model')
    return loaded_model

def loadMfcc3dCnnModel():
    loaded_model = loadKerasModel('ceps_cnn3d_model')
    return loaded_model

