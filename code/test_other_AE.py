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

    bas = []
    params = []

    # data parameters
    Y = sio.loadmat("../data/" + filename)
    y_all = Y["X"]
    y_truth_all = Y["y"]
    y_truth_all = y_truth_all.astype(np.int16)
    y_truth_all = y_truth_all * (-2) + 1
    nr_out = y_truth_all[y_truth_all == -1].shape[0]
    print(f"\n         {filename} {y_all.shape}, {nr_out} outliers")

    dataset = filename.split(".")[0]
    y_all = StandardScaler().fit_transform(y_all)

    for optimizer in optimizers:
        for activation in activations:
            for dropout_rate in dropout_rates:

                autoencoder = load_AE_config("../init/AE-" + dataset + ".yaml", nu[filename])
                autoencoder.contamination = nu[filename]

                autoencoder.optimizer = optimizer
                autoencoder.activation_function = activation
                autoencoder.dropout = dropout_rate

                start_time = time.time()
                autoencoder.fit(y_all)
                train_time = time.time() - start_time

                start_time = time.time()
                predictions = autoencoder.predict(y_all)
                test_time = time.time() - start_time

                predictions = predictions * (-2) + 1
                sensitivity, specificity, acc = fpfn(predictions, y_truth_all)
                ba = (sensitivity + specificity) / 2

                bas.append(ba)
                params.append((optimizer, activation, dropout_rate))

                print(f'AE: sensitivity {sensitivity}, specificity {specificity} BA {ba}')
                print(f'Train time: {train_time}, Test time: {test_time}')

    max_index = np.argmax(bas)
    print(f"Max BA is {np.max(bas)}, obtained for {params[max_index]}")
