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
import sys
import time
import warnings

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.model_selection import KFold, train_test_split
from utils import fpfn, load_AE_config

warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


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

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    filename = sys.argv[1]

    if filename not in nu.keys():
        print("Invalid dataset name! You have to specify one of these: " + str(list(nu.keys())))
        exit(0)

    bas_all = []
    params_all = []

    # data parameters
    Y = sio.loadmat("../data/" + filename)
    y_all = Y["X"]
    y_truth_all = Y["y"]
    y_truth_all = y_truth_all.astype(np.int16)
    y_truth_all = y_truth_all * (-2) + 1
    nr_out = y_truth_all[y_truth_all == -1].shape[0]
    print(f"\n         {filename} {y_all.shape}, {nr_out} outliers")

    dataset = filename.split(".")[0]

    Y_train, Y_test, y_truth_train, y_truth_test = train_test_split(
        y_all, y_truth_all, test_size=0.2, random_state=39
    )

    for optimizer in optimizers:
        for activation in activations:
            for dropout_rate in dropout_rates:

                kf = KFold(n_splits=5)
                kf.get_n_splits(Y_train)
                bas = []
                sens = []
                specs = []
                train_times = []
                test_times = []
                for i, (train_index, test_index) in enumerate(kf.split(Y_train)):
                    Y_train_folds = Y_train[train_index]
                    Y_train_folds = StandardScaler().fit_transform(Y_train_folds)
                    Y_test_fold = Y_train[test_index]
                    Y_test_fold = StandardScaler().fit_transform(Y_test_fold)

                    y_test_truth = y_truth_train[test_index]

                    autoencoder = load_AE_config("../init/AE-" + dataset + ".yaml", nu[filename])
                    autoencoder.contamination = nu[filename]

                    autoencoder.optimizer = optimizer
                    autoencoder.activation_function = activation
                    autoencoder.dropout = dropout_rate

                    start_time = time.time()
                    autoencoder.fit(Y_train_folds)
                    train_time = time.time() - start_time

                    start_time = time.time()
                    predictions = autoencoder.predict(Y_test_fold)
                    test_time = time.time() - start_time

                    predictions = predictions * (-2) + 1
                    sensitivity, specificity, acc = fpfn(predictions, y_test_truth)
                    ba = (sensitivity + specificity) / 2

                    bas.append(ba)
                    sens.append(sensitivity)
                    specs.append(specificity)
                    train_times.append(train_time)
                    test_times.append(test_time)

                bas_all.append(np.mean(bas))
                params_all.append((optimizer, activation, dropout_rate))

    max_index = np.argmax(bas_all)
    print(f"Max BA is {np.max(bas)} and it is obtained for {params_all[max_index]}")

    autoencoder = load_AE_config("../init/AE-" + dataset + ".yaml", nu[filename])
    autoencoder.contamination = nu[filename]

    autoencoder.optimizer = params_all[max_index][0]
    autoencoder.activation_function = params_all[max_index][1]
    autoencoder.dropout = params_all[max_index][2]

    Y_train = StandardScaler().fit_transform(Y_train)
    Y_test = StandardScaler().fit_transform(Y_test)
    autoencoder.fit(Y_train)

    predictions = autoencoder.predict(Y_test)
    predictions = predictions * (-2) + 1
    sensitivity, specificity, acc = fpfn(predictions, y_truth_test)
    ba = (sensitivity + specificity) / 2
    print(f'BA on test set is {ba}')
