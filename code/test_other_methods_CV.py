# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os
import warnings

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from utils import fpfn

warnings.simplefilter("ignore", DataConversionWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


def evaluate_clf(clf, grid_param, Y_train, y_train):
    grid = GridSearchCV(estimator=clf,
                        param_grid=grid_param,
                        scoring='balanced_accuracy',
                        cv=5,
                        n_jobs=4
                        )

    grid.fit(Y_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_, grid.refit_time_


def evaluate_ocsvm(Y_train, y_train, nu=None):
    grid_param = {
        'gamma':  ['auto', 'scale', 1/16654, 1/8126, 1/4096, 1/2048, 1/1024, 1/256, 1/128, 1/64, 1/32, 1/12, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'max_iter': [5000]
    }

    if nu is not None:
        grid_param['nu'] = [nu]

    ocsvm = OneClassSVM()

    return evaluate_clf(ocsvm, grid_param, Y_train, y_train)


def evaluate_lof(Y_train, y_train, contamination=None):
    n_neighbors = range(3, 14, 2)
    leaf_size = range(20, 60, 10)
    grid_param = {
        'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                   'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
                   'hamming', 'jaccard', 'minkowski', 'rogerstanimoto',
                   'russellrao'],
        'n_neighbors': n_neighbors,
        'leaf_size': leaf_size,
        'p': [1, 2, 5, 50],
        'novelty': [True]
    }

    if contamination is not None:
        grid_param['contamination'] = [contamination]

    lof = LocalOutlierFactor()

    return evaluate_clf(lof, grid_param, Y_train, y_train)


def evaluate_iforest(Y_train, y_train, contamination=None):
    n_estimators = range(50, 450, 50)
    grid_param = {
        'n_estimators': n_estimators,
        'bootstrap': [True, False]
    }

    if contamination is not None:
        grid_param['contamination'] = [contamination]

    iforest = IsolationForest()

    return evaluate_clf(iforest, grid_param, Y_train, y_train)


if __name__ == "__main__":
    nu = {
        "satellite.mat": 0.317,
        "mnist.mat": 0.093,
        "shuttle.mat": 0.072,
        "pendigits.mat": 0.023,
        "speech.mat": 0.017,
        "cardio.mat": 0.096,
        "glass.mat": 0.042,
        "thyroid.mat": 0.025,
        "vowels.mat": 0.034,
        "wbc.mat": 0.056,
        "lympho.mat": 0.041,
    }

    for key in nu.keys():
        filename = key
        print("\n         %s" % filename)

        Y = sio.loadmat("../data/" + filename)
        y_all = Y["X"]
        y_truth_all = Y["y"]
        y_truth_all = y_truth_all.astype(np.int16)
        y_truth_all = y_truth_all * (-2) + 1

        Y_train, Y_test, y_truth_train, y_truth_test = train_test_split(
            y_all, y_truth_all, test_size=0.2, random_state=33
        )

        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y_train)

        Y_test = scaler.transform(Y_test)

        best_clf, best_score, best_params, refit_time = evaluate_ocsvm(Y_train, y_truth_train, nu[filename])
        print("Best Score for OCSVM on validation data is :%s" % best_score)
        print("Best params: %s" % best_params)

        best_clf.fit(Y_train)
        y_pred_test = best_clf.predict(Y_test)
        sensitivity, specificity, accuracy = fpfn(y_pred_test, y_truth_test)
        print("Score on test data is :%s" % str((sensitivity + specificity) / 2))

        best_clf, best_score, best_params, refit_time = evaluate_iforest(Y_train, y_truth_train, nu[filename])
        print("Best Score for IFOREST on validation data is :%s" % best_score)
        print("Best params: %s" % best_params)

        best_clf.fit(Y_train)
        y_pred_test = best_clf.predict(Y_test)
        sensitivity, specificity, accuracy = fpfn(y_pred_test, y_truth_test)
        print("Score on test data is :%s" % str((sensitivity + specificity) / 2))

        best_clf, best_score, best_params, refit_time = evaluate_lof(Y_train, y_truth_train, nu[filename])
        print("Best Score for LOF on validation data is :%s" % best_score)
        print("Best params: %s" % best_params)

        best_clf.fit(Y_train)
        y_pred_test = best_clf.predict(Y_test)
        sensitivity, specificity, accuracy = fpfn(y_pred_test, y_truth_test)
        print("Score on test data is :%s" % str((sensitivity + specificity) / 2))
