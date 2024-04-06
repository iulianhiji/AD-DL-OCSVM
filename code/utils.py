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

import pickle

# from attrdict import AttrDict
import attridict
import numpy as np
import yaml
from graphomaly.models.autoencoder import Autoencoder
from sklearn.metrics import confusion_matrix


def fpfn(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, -1]).ravel()
    if tp == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    if tn == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    acc = (tn + tp) / (tn + tp + fn + fp)
    return sensitivity, specificity, acc


def test_fpfn_DL_ocsvm_representations(X_test, y_true, fitted_clf):
    clf = fitted_clf
    y_pred = clf.predict(X_test.T)
    return fpfn(y_pred, y_true)


def test_fpfn_DPL_ocsvm_representations(X_test, X_test_2, y_true, fitted_clf):
    clf = fitted_clf
    y_pred = clf.predict(X_test.T)
    y_pred2 = clf.predict(X_test_2.T)
    return fpfn(y_pred, y_true) + fpfn(y_pred2, y_true)


def load_dicts(filename, kernel=False):
    dicts = []
    initFile = "../init/DL-NN-init-"
    if kernel is True:
        initFile += "kernel-"
    with open(initFile + filename.split(".")[0] + ".pickle", "rb") as infile:
        dicts = np.array(pickle.load(infile))
    return dicts


def generate_dicts(nr_dicts, n, p, filename, kernel=False):
    dicts = []
    for i in range(nr_dicts):
        D = np.random.rand(n, p)
        D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D
        dicts.append(D)
    initFile = "../init/DL-NN-init-"
    if kernel is True:
        initFile += "kernel-"
    with open(initFile + filename.split(".")[0] + ".pickle", "wb") as outfile:
        pickle.dump(dicts, outfile)
    print("Dictionaries were saved!")


def load_AE_config(config_file, contamination):
    with open(config_file) as f:
        config = attridict(yaml.safe_load(f))
        config.model["AmlAE"]["contamination"] = contamination
        autoencoder = Autoencoder()
        autoencoder.set_params(**config.model["AmlAE"])
        return autoencoder
