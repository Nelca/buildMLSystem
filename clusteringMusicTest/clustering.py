# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.metrics import confusion_matrix


from read_dataset import read_ceps


GENRE_LIST = ["blues", "classical", "jazz", "country", "pop", "rock", "metal"]
TEST_DIR = "/media/sf_P/pymlbook-data/09-genre-class/private"

genre_list = GENRE_LIST


def train_model(clf_factory, X, Y, name):
    labels = np.unique(Y)

    cv = ShuffleSplit(n=len(X), n_iter=1, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    pr_scores_list = np.array([])

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

            precision, recall, pr_thresholds = precision_recall_curve(y_label_test, proba_label)
            auc_result = auc(recall, precision)
            pr_scores[label].append(auc(recall, precision))
            pr_scores_list = np.append(pr_scores_list, auc_result)

    summary = (np.mean(scores), np.std(scores), np.mean(pr_scores_list), np.std(pr_scores_list))
    print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


def create_model():
    clf = LogisticRegression()

    return clf


if __name__ == "__main__":
    X, y = read_ceps(genre_list)

    train_avg, test_avg, cms = train_model(create_model, X, y, "Log Reg CEPS")

    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)

    print(cm_norm)

