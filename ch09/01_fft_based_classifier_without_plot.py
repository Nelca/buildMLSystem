# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import os
import glob
from collections import defaultdict

import pdb

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import confusion_matrix
GENRE_LIST = ["classical", "jazz", "country", "pop", "rock", "metal"]

DATA_DIR = os.path.join("..", "data/songData/")
CHART_DIR = os.path.join("..", "charts")

GENRE_DIR = "../data/songData/genres/"

TEST_DIR = "/media/sf_P/pymlbook-data/09-genre-class/private"
genre_list = GENRE_LIST

#GENRE_DIR = "/media/sf_P/pymlbook-data/09-genre-class/genres"
#from fft import read_fft


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

def train_model(clf_factory, X, Y, name):
    labels = np.unique(Y)

    cv = ShuffleSplit(
        n=len(X), n_iter=1, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(
        list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = []  # just to later get the median

    cms = []

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = clf_factory()
        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

        for label in labels:
            y_label_test = np.asarray(y_test == label, dtype=int)
            proba = clf.predict_proba(X_test)
            proba_label = proba[:, label]

            precision, recall, pr_thresholds = precision_recall_curve(
                y_label_test, proba_label)
            pr_scores[label].append(auc(recall, precision))
            precisions[label].append(precision)
            recalls[label].append(recall)
            thresholds[label].append(pr_thresholds)

            fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
            roc_scores[label].append(auc(fpr, tpr))
            tprs[label].append(tpr)
            fprs[label].append(fpr)

    all_pr_scores = np.asarray(pr_scores.values()).flatten()
    pdb.set_trace()
    all_pr_scores_dictval = all_pr_scores[0]
    np_m_sc = np.mean(scores)
    np_std_sc = np.std(scores)
    np_m_all_sc = np.mean(all_pr_scores_dictval)
    np_std_all_sc = np.std(all_pr_scores_dictval)
    summary = (np_m_sc, np_std_sc,
               np_m_all_sc, np_std_all_sc)
    print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


def create_model():
    from sklearn.linear_model.logistic import LogisticRegression
    clf = LogisticRegression()

    return clf


if __name__ == "__main__":
    X, y = read_fft(genre_list)

    train_avg, test_avg, cms = train_model(
        create_model, X, y, "Log Reg FFT")

    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)

    print(cm_norm)

