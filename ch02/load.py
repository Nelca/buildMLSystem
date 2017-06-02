# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import pdb

def load_dataset(dataset_name):
    '''
    data,labels = load_dataset(dataset_name)

    Load a given dataset

    Returns
    -------
    data : numpy ndarray
    labels : list of str
    '''
    data = []
    labels = []
    file_path = '../data/{0}.tsv'.format(dataset_name)
    #file_data = open(file_path)
    #for ifile in file_data:
    pdb.set_trace()
    with open(file_path) as ifile:
        f_data = ifile.read_all()
        #for line in ifile:
        for line in f_data:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
