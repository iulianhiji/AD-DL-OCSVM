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
from cmath import sqrt

import dictlearn as dl
import numpy as np
import scipy
import scipy.io as sio
from dictlearn import DictionaryLearning
from joblib import Parallel, delayed
from ksvd_supp import ksvd_supp_DPL_OCSVM
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import OneClassSVM
from utils import load_dicts, test_fpfn_DL_ocsvm_representations


def dl_ocsvm_predict(Y, D, y_truth, params, n_nonzero_coefs, use_OMP):
    start_time = time.time()

    if use_OMP is True:
        X = dl._dictionary_learning.sparse_encode(
            Y, D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
        )[0]
        P = params["P"]

        for atom_index in range(D.shape[1]):
            if np.all(P[atom_index, :] == 0):
                X[atom_index, :] = np.zeros_like(X[atom_index, :])

    else:
        X = params["P"] @ Y

    ocsvm = params["clf"]
    sensitivity, specificity, acc = test_fpfn_DL_ocsvm_representations(
        X, y_truth, ocsvm
    )

    elapsed_time = time.time() - start_time

    return (sensitivity + specificity) / 2, sensitivity, specificity, elapsed_time


def ksvd_update(beta, gamma, filename, nu, Y, y_truth_all, dictionary, n_nonzero_coefs,
                use_OMP_for_prediction):
    # Initialization

    n_features = np.shape(Y)[0]  # signal dimension
    n_components = 2 * n_features  # number of atoms

    params = {
        "replatoms": 2,
        "atom_norm_tolerance": 1e-10,
        "supp_reg": 1,
        "supp_beta": beta,
    }  # NO

    params["nu"] = nu

    # generate_dicts(20, n, n_components, filename)
    # exit(0)

    D = dictionary
    D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

    start_time = time.time()

    X = dl._dictionary_learning.sparse_encode(
        Y, D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
    )[0]

    clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)
    params["clf"] = clf
    params["u"] = clf.coef_[0] / (clf.nu * X.T.shape[0])
    params["lambda"] = np.zeros(X.T.shape[0])
    params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

    norm_0 = np.linalg.norm(X, axis=1, ord=0)
    alpha = norm_0 / np.linalg.norm(norm_0)
    params["alpha"] = alpha
    params["gamma"] = gamma

    # Introduce P
    P = np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T
    params["P"] = P
    params["Y"] = Y

    n_iterations = 6  # number of DL iterations (K)

    dlearn = DictionaryLearning(
        n_components=n_components,
        max_iter=n_iterations,
        fit_algorithm=ksvd_supp_DPL_OCSVM,
        n_nonzero_coefs=n_nonzero_coefs,
        dict_init=D,
        params=params,
        data_sklearn_compat=False,
        transform_algorithm=None,
        code_init=X,
    )

    dlearn.fit(Y)

    dictionary = dlearn.D_
    code = dlearn.X_
    # print(code[0])

    sensitivity, specificity, acc = test_fpfn_DL_ocsvm_representations(
        code, y_truth_all, params["clf"]
    )

    elapsed_time = time.time() - start_time

    balanced_accuracy = (sensitivity + specificity) / 2

    balanced_accuracy2, sensitivity2, specificity2, _elapsed_time = dl_ocsvm_predict(
        Y,
        dictionary,
        y_truth_all,
        params,
        n_nonzero_coefs,
        use_OMP_for_prediction
    )

    print(
        "BETA: %s, GAMMA: %s,  BA: %s, BA2: %s"
        % (beta, gamma, balanced_accuracy, balanced_accuracy2)
    )

    return balanced_accuracy, sensitivity, specificity, elapsed_time


def Kfold_cv(D, Y_train, y_test, n_nonzero_coefs, params, use_OMP_for_prediction):
    # We already have Y.T, so we have to transpose again
    Y_train = Y_train.T

    kf = KFold(n_splits=5)
    kf.get_n_splits(Y_train)
    bas = []
    sens = []
    specs = []
    train_times = []
    test_times = []
    for i, (train_index, test_index) in enumerate(kf.split(Y_train)):
        Y_train_folds = Y_train[train_index]
        Y_test_fold = Y_train[test_index]

        y_test_fold = y_test[test_index]

        Y_train_folds = Y_train_folds.T
        Y_test_fold = Y_test_fold.T

        start_time = time.time()

        X = dl._dictionary_learning.sparse_encode(
            Y_train_folds, D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
        )[0]

        norm_0 = np.linalg.norm(X, axis=1, ord=0)
        alpha = norm_0 / np.linalg.norm(norm_0)
        params["alpha"] = alpha

        # Introduce P
        P = np.linalg.lstsq(Y_train_folds.T, X.T, rcond=None)[0].T
        params["P"] = P
        params["Y"] = Y_train_folds

        n_iterations = 6  # number of DL iterations (K)
        n_features = np.shape(Y_train_folds)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms

        clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)
        params["clf"] = clf
        params["u"] = clf.coef_[0] / (clf.nu * X.T.shape[0])
        params["lambda"] = np.zeros(X.T.shape[0])
        params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

        dlearn = DictionaryLearning(
            n_components=n_components,
            max_iter=n_iterations,
            fit_algorithm=ksvd_supp_DPL_OCSVM,
            n_nonzero_coefs=n_nonzero_coefs,
            dict_init=D,
            params=params,
            data_sklearn_compat=False,
            transform_algorithm=None,
            code_init=X,
        )

        dlearn.fit(Y_train_folds)

        elapsed_time = time.time() - start_time

        dictionary = dlearn.D_

        ba, sensitivity, specificity, test_time = dl_ocsvm_predict(
            Y_test_fold,
            dictionary,
            y_test_fold,
            params,
            n_nonzero_coefs,
            use_OMP_for_prediction
        )
        bas.append(ba)
        sens.append(sensitivity)
        specs.append(specificity)
        train_times.append(elapsed_time)
        test_times.append(test_time)

    balanced_accuracy = np.mean(bas)
    sensitivity = np.mean(sens)
    specificity = np.mean(specs)
    train_time = np.mean(train_times)
    test_time = np.mean(test_times)
    print(
        "BETA: %s, GAMMA: %s  BA: %s" % (params["supp_beta"],
                                         params["gamma"], balanced_accuracy)
    )
    return balanced_accuracy, sensitivity, specificity, train_time, test_time


def ksvd_update_Kfold_cv(beta, gamma, filename, nu, Y, y_truth_test, dictionary,
                         n_nonzero_coefs, use_OMP_for_prediction):
    Y = Y.T
    Y_norm = np.linalg.norm(Y, ord="fro")
    Y = Y / Y_norm

    # Initialization
    m = Y.shape[1]

    n_features = np.shape(Y)[0]  # signal dimension
    n_components = 2 * n_features  # number of atoms

    params = {
        "replatoms": 2,
        "atom_norm_tolerance": 1e-10,
        "supp_reg": 1,
        "supp_beta": beta,
    }  # NO

    u = np.ones(n_components) / n_components
    params["u"] = u
    params["lambda"] = np.ones(m) / sqrt(m).real
    params["nu"] = nu
    params["gamma"] = gamma

    D = dictionary
    D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

    return Kfold_cv(D, Y, y_truth_test, n_nonzero_coefs, params, use_OMP_for_prediction)


def run_experiment(dir, filename, Y, y_truth_all, beta, nu, cv, n_nonzero_coefs,
                   use_OMP_for_prediction):
    # for gamma in np.geomspace(1e-4, 1e-2, 3):  # np.linspace(0.05, 0.5, 10):
    for gamma in np.linspace(0.05, 0.5, 10):
        gamma = round(gamma, 16)

        bas = []
        tnrs = []
        tprs = []

        elapsed_times = []

        dicts = load_dicts(filename)

        for i in range(20):
            if cv is False:
                ba, tpr, tnr, train_time = ksvd_update(
                    beta,
                    gamma,
                    filename,
                    nu[filename],
                    Y, y_truth_all,
                    dicts[i],
                    n_nonzero_coefs,
                    use_OMP_for_prediction
                )
            else:
                ba, tpr, tnr, train_time, test_time = ksvd_update_Kfold_cv(
                    beta,
                    gamma,
                    filename,
                    nu[filename],
                    Y,
                    y_truth_all,
                    dicts[i],
                    n_nonzero_coefs,
                    use_OMP_for_prediction
                )
            bas.append(ba)
            tprs.append(tpr)
            tnrs.append(tnr)
            elapsed_times.append(train_time)

        max_ba_beta = np.max(bas)

        max_ba_index = bas.index(max_ba_beta)

        mean_ba_beta = np.mean(bas)
        std_ba_beta = np.std(bas)

        mean_time = np.mean(elapsed_times)

        scipy.io.savemat(
            dir + '/' + f'OCSVM-DPL-TR_beta{str(beta)}_gamma{str(gamma)}.mat',
            {'BA': max_ba_beta,
             'meanBA': mean_ba_beta,
             'stdBA': std_ba_beta,
             'TPR': tprs[max_ba_index],
             'TNR': tnrs[max_ba_index],
             'time': elapsed_times[max_ba_index],
             'mean-time': mean_time,
             'dict': dicts[max_ba_index]})


def compare_results(dir):
    max_ba = 0
    max_beta_gamma = None
    max_ba_beta_dict = {}
    for path in os.listdir(dir):
        data = scipy.io.loadmat(os.path.join(dir, path))
        name = path.split("OCSVM-DPL-TR_", 1)[1]
        name = name.split(".mat", 1)[0]
        max_ba_beta_dict[name] = data  # max_ba_beta

    for key in max_ba_beta_dict.keys():
        if max_ba_beta_dict[key]['BA'] > max_ba:
            max_ba = max_ba_beta_dict[key]['BA']
            max_beta_gamma = key

    print("MAX BA: %s (tpr: %s, tnr: %s) mean: %s, std: %s OBTAINED FOR BETA and GAMMA: %s, time: %s (mean: %s)" % (
        max_ba, max_ba_beta_dict[max_beta_gamma]['TPR'], max_ba_beta_dict[max_beta_gamma]['TNR'],
        max_ba_beta_dict[max_beta_gamma]['meanBA'], max_ba_beta_dict[max_beta_gamma]['stdBA'],
        max_beta_gamma, max_ba_beta_dict[max_beta_gamma]['time'],
        max_ba_beta_dict[max_beta_gamma]['mean-time'])
    )

    beta_tmp = max_beta_gamma.split("beta", 1)[1]
    max_beta = beta_tmp.split("_gamma", 1)[0]
    max_gamma = max_beta_gamma.split("_gamma", 1)[1]

    return max_ba_beta_dict[max_beta_gamma]['dict'], max_beta, max_gamma


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

    n_nonzero_coef = {
        "satellite.mat": 6,
        "mnist.mat": 8,
        "shuttle.mat": 4,
        "pendigits.mat": 5,
        "speech.mat": 10,
        "cardio.mat": 5,
        "glass.mat": 3,
        "thyroid.mat": 3,
        "vowels.mat": 4,
        "wbc.mat": 6,
        "lympho.mat": 4,
    }

    cv = False

    use_OMP_for_prediction = True

    filename = sys.argv[1]
    # filename = "lympho.mat"

    if filename not in nu.keys():
        print("Invalid dataset name! You have to specify one of these: " + str(list(nu.keys())))
        exit(0)

    betas = np.linspace(0.01, 0.5, 50)  # np.geomspace(1e-4, 1, 5)

    if cv is False:
        dir = "results/" + filename.split(".mat")[0] + "_DPL6iter"
    else:
        dir = "results/" + filename.split(".mat")[0] + "_DPL_KFold"
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, dir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # data parameters
    Y = sio.loadmat("../data/" + filename)
    y_all = Y["X"]
    y_truth_all = Y["y"]
    y_truth_all = y_truth_all.astype(np.int16)
    y_truth_all = y_truth_all * (-2) + 1

    if cv is False:
        Y = y_all.T
        Y_norm = np.linalg.norm(Y, ord="fro")
        Y = Y / Y_norm

        Parallel(n_jobs=5)(delayed(run_experiment)(
            dir,
            filename,
            Y, y_truth_all,
            round(beta, 16),
            nu,
            cv,
            n_nonzero_coef[filename],
            use_OMP_for_prediction) for beta in betas
        )
    else:
        Y_train, Y_test, y_truth_train, y_truth_test = train_test_split(
            y_all, y_truth_all, test_size=0.2, random_state=37
        )
        Parallel(n_jobs=5)(delayed(run_experiment)(
            dir,
            filename,
            Y_train,
            y_truth_train,
            round(beta, 16),
            nu,
            cv,
            n_nonzero_coef[filename],
            use_OMP_for_prediction) for beta in betas
        )

    D, max_beta, max_gamma = compare_results(final_directory)
    max_beta = float(max_beta)
    max_gamma = float(max_gamma)
    print(f"beta: {max_beta}, gamma: {max_gamma}")

    if cv is True:
        # Specific for K Fold Validation
        D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

        n_iterations = 6  # number of DL iterations (K)
        n_features = np.shape(Y_train.T)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms

        params = {
            "replatoms": 2,
            "atom_norm_tolerance": 1e-10,
            "supp_reg": 1,
            "supp_beta": max_beta,
        }  # NO

        params["nu"] = nu[filename]

        Y_norm = np.linalg.norm(Y_train, ord="fro")
        Y_train = Y_train / Y_norm

        Y_norm = np.linalg.norm(Y_test, ord="fro")
        Y_test = Y_test / Y_norm

        X = dl._dictionary_learning.sparse_encode(
            Y_train.T, D, algorithm="omp", n_nonzero_coefs=n_nonzero_coef[filename]
        )[0]

        clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)
        params["clf"] = clf
        params["u"] = clf.coef_[0] / (clf.nu * X.T.shape[0])
        params["lambda"] = np.zeros(X.T.shape[0])
        params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

        norm_0 = np.linalg.norm(X, axis=1, ord=0)
        alpha = norm_0 / np.linalg.norm(norm_0)
        params["alpha"] = alpha
        params["gamma"] = max_gamma

        # Introduce P
        P = np.linalg.lstsq(Y_train, X.T, rcond=None)[0].T
        params["P"] = P
        params["Y"] = Y_train.T

        dlearn = DictionaryLearning(
            n_components=n_components,
            max_iter=n_iterations,
            fit_algorithm=ksvd_supp_DPL_OCSVM,
            n_nonzero_coefs=n_nonzero_coef[filename],
            dict_init=D,
            params=params,
            data_sklearn_compat=False,
            transform_algorithm=None,
            code_init=X,
        )

        dlearn.fit(Y_train.T)

        ba, sensitivity, specificity, elapsed_time = dl_ocsvm_predict(
            Y_train.T,
            dlearn.D_,
            y_truth_train,
            params,
            n_nonzero_coef[filename],
            use_OMP_for_prediction
        )
        print("BA on entire TRAIN set: %s" % str(ba))

        ba, sensitivity, specificity, elapsed_time = dl_ocsvm_predict(
            Y_test.T,
            dlearn.D_,
            y_truth_test,
            params,
            n_nonzero_coef[filename],
            use_OMP_for_prediction
        )
        print("BA on test set: %s" % str(ba))

    # Else JUST PREDICT USING THE 2 VARIANTS (WITH AND WITHOUT USING OMP)
    else:
        # Initialization
        Y = y_all.T
        Y_norm = np.linalg.norm(Y, ord="fro")
        Y = Y / Y_norm
        m = Y.shape[1]

        n_features = np.shape(Y)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms

        params = {
            "replatoms": 2,
            "atom_norm_tolerance": 1e-10,
            "supp_reg": 1,
            "supp_beta": max_beta,
        }

        D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

        start_time = time.time()

        X = dl._dictionary_learning.sparse_encode(
            Y, D, algorithm="omp", n_nonzero_coefs=n_nonzero_coef[filename]
        )[0]

        clf = OneClassSVM(kernel="linear", nu=nu[filename], max_iter=5000).fit(X.T)
        params["clf"] = clf
        params["u"] = clf.coef_[0] / (clf.nu * X.T.shape[0])
        params["lambda"] = np.zeros(X.T.shape[0])
        params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])
        params["nu"] = nu[filename]

        norm_0 = np.linalg.norm(X, axis=1, ord=0)
        alpha = norm_0 / np.linalg.norm(norm_0)
        params["alpha"] = alpha
        params["gamma"] = max_gamma

        # Introduce P
        P = np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T
        params["P"] = P
        params["Y"] = Y

        n_iterations = 6  # number of DL iterations (K)
        dlearn = DictionaryLearning(
            n_components=n_components,
            max_iter=n_iterations,
            fit_algorithm=ksvd_supp_DPL_OCSVM,
            n_nonzero_coefs=n_nonzero_coef[filename],
            dict_init=D,
            params=params,
            data_sklearn_compat=False,
            transform_algorithm=None,
            code_init=X,
        )

        dlearn.fit(Y)
        sensitivity, specificity, acc = test_fpfn_DL_ocsvm_representations(
            dlearn.X_, y_truth_all, params["clf"]
        )

        ba = (sensitivity + specificity) / 2
        print("BA on entire set (for resulted representations): %s" % str(ba))
