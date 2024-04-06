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
import time
import warnings

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from utils import fpfn

warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


def test_fpfn_ocsvm(X, y_true, gamma, kernel):
    clf = make_pipeline(StandardScaler(), OneClassSVM(gamma=gamma, kernel=kernel, max_iter=5000))
    start_time = time.time()
    y_pred = clf.fit_predict(X)
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X)
    test_time = time.time() - start_time
    sensitivity, specificity, acc = fpfn(y_pred, y_true)
    return sensitivity, specificity, acc, train_time, test_time


def test_fpfn_lof(X, y_true, n_neighbors, leaf_size, metric, p):
    clf = make_pipeline(StandardScaler(), LocalOutlierFactor(
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        metric=metric, p=p)
    )
    start_time = time.time()
    y_pred = clf.fit_predict(X)
    train_time = time.time() - start_time
    sensitivity, specificity, acc = fpfn(y_pred, y_true)
    return sensitivity, specificity, acc, train_time


def test_fpfn_iforrest(X, y_true, n_estimators, bootstrap, max_features):
    clf = make_pipeline(StandardScaler(), IsolationForest(
        random_state=0,
        n_estimators=n_estimators,
        bootstrap=bootstrap,
        max_features=max_features,
        warm_start=True)
    )
    start_time = time.time()
    y_pred = clf.fit_predict(X)
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X)
    test_time = time.time() - start_time
    sensitivity, specificity, acc = fpfn(y_pred, y_true)
    return sensitivity, specificity, acc, train_time, test_time


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

        Y = sio.loadmat("../data/" + filename)
        y_all = Y["X"]
        y_truth_all = Y["y"]
        y_truth_all = y_truth_all.astype(np.int16)
        y_truth_all = y_truth_all * (-2) + 1
        nr_out = y_truth_all[y_truth_all == -1].shape[0]
        print(f"\n         {filename} {y_all.shape}, {nr_out} outliers")

        values = []
        sensitivity_if = []
        specificity_if = []
        train_time_if = []
        test_time_if = []

        for n_estimators in range(50, 450, 50):
            for bootstrap in [True, False]:
                for max_features in range(1, np.shape(y_all)[1] + 1, 2):
                    sensitivity, specificity, acc, train_time, test_time = test_fpfn_iforrest(
                        y_all,
                        y_truth_all,
                        n_estimators=n_estimators,
                        bootstrap=bootstrap,
                        max_features=max_features
                    )
                    sensitivity_if.append(sensitivity)
                    specificity_if.append(specificity)
                    train_time_if.append(train_time)
                    test_time_if.append(test_time)
                    values.append((sensitivity + specificity) / 2)

        maxIndex = np.argmax(values)
        sensitivity = sensitivity_if[maxIndex]
        specificity = specificity_if[maxIndex]
        train_time = train_time_if[maxIndex]
        test_time = test_time_if[maxIndex]
        train_time_mean = np.mean(train_time_if)
        test_time_mean = np.mean(test_time_if)
        ba = values[maxIndex]
        print(f'IForrest: sensitivity {sensitivity}, specificity {specificity}, BA {ba}, Train time {train_time}(mean: {train_time_mean}, Test time: {test_time}(mean: {test_time_mean})) ')

        sensitivity_lof = []
        specificity_lof = []
        values = []
        train_time_lof = []

        metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                   'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
                   'hamming', 'jaccard', 'minkowski', 'rogerstanimoto',
                   'russellrao']
        for n_neighbors in range(3, 14, 2):
            for leaf_size in range(20, 60, 10):
                for metric in metrics:
                    for p in [1, 2, 5, 50]:
                        sensitivity, specificity, acc, train_time = test_fpfn_lof(
                            y_all,
                            y_truth_all,
                            n_neighbors,
                            leaf_size,
                            metric,
                            p
                        )
                        sensitivity_lof.append(sensitivity)
                        specificity_lof.append(specificity)
                        train_time_lof.append(train_time)
                        values.append((sensitivity + specificity) / 2)

        maxIndex = np.argmax(values)
        sensitivity = sensitivity_lof[maxIndex]
        specificity = specificity_lof[maxIndex]
        train_time = train_time_lof[maxIndex]
        train_time_mean = np.mean(train_time_lof)
        ba = values[maxIndex]
        print(f'LOF: sensitivity {sensitivity}, specificity {specificity} BA {ba}, Train time {train_time}(mean: {train_time_mean})) ')

        gammas = ['auto', 'scale', 1/16654, 1/8126, 1/4096, 1/2048, 1/1024, 1/256,
                  1/128, 1/64, 1/32, 1/12, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        values = []
        sensitivity_ocsvm = []
        specificity_ocsvm = []
        train_time_ocsvm = []
        test_time_ocsvm = []

        for gamma in gammas:
            for kernel in kernels:
                sensitivity, specificity, acc, train_time, test_time = test_fpfn_ocsvm(
                    y_all,
                    y_truth_all,
                    gamma,
                    kernel
                )
                sensitivity_ocsvm.append(sensitivity)
                specificity_ocsvm.append(specificity)
                train_time_ocsvm.append(train_time)
                test_time_ocsvm.append(test_time)
                values.append((sensitivity + specificity)/2)

        maxIndex = np.argmax(values)
        sensitivity = sensitivity_ocsvm[maxIndex]
        specificity = specificity_ocsvm[maxIndex]
        train_time = train_time_ocsvm[maxIndex]
        test_time = test_time_ocsvm[maxIndex]
        train_time_mean = np.mean(train_time_ocsvm)
        test_time_mean = np.mean(test_time_ocsvm)

        ba = values[maxIndex]

        print(f'OCSVM: sensitivity {sensitivity}, specificity {specificity} BA {ba}, Train time {train_time}(mean: {train_time_mean}, Test time: {test_time}(mean: {test_time_mean}))')
