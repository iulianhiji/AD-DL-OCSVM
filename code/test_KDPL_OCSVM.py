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
from ksvd_supp import ker_ksvd_supp_DPL_OCSVM
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import OneClassSVM
from utils import load_dicts, test_fpfn_DL_ocsvm_representations


def dl_kernel_DPL_predict(Y, Y_train, K_train, D, y_truth, params,
                          n_nonzero_coefs, gamma_ker):
    start_time = time.time()

    # precompute the Kernel matrix, it s sq root, and sq root's pinv
    K = rbf_kernel(Y_train.T, Y.T, gamma_ker)

    X, _err = dl._dictionary_learning.sparse_encode(
        D.T @ K, D.T @ K_train @ D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
    )

    B = params["B"]

    for atom_index in range(D.shape[1]):
        if np.all(B[atom_index, :] == 0):
            X[atom_index, :] = np.zeros_like(X[atom_index, :])

    ocsvm = params["clf"]
    sensitivity, specificity, acc = test_fpfn_DL_ocsvm_representations(
        X, y_truth, ocsvm
    )

    elapsed_time = time.time() - start_time

    return (sensitivity + specificity) / 2, sensitivity, specificity, elapsed_time


def ksvd_update(beta, gamma, filename, nu, Y, y_truth_all, dictionary, n_nonzero_coefs,
                K, K_sqr, K_sqr_inv):
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

    # generate_dicts(20, Y.shape[1], n_components, filename, kernel=True)
    # exit(0)

    D = dictionary
    D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

    start_time = time.time()

    X = dl._dictionary_learning.sparse_encode(
        D.T @ K, D.T @ K @ D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
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

    # Introduce B
    B = np.linalg.lstsq(K, X.T, rcond=None)[0].T
    params["B"] = B

    n_iterations = 6  # number of DL iterations (K)

    params["K"] = K
    params["K_sqr"] = K_sqr
    params["K_sqr_inv"] = K_sqr_inv

    dlearn = DictionaryLearning(
        n_components=n_components,
        max_iter=n_iterations,
        fit_algorithm=ker_ksvd_supp_DPL_OCSVM,
        n_nonzero_coefs=n_nonzero_coefs,
        dict_init=D,
        params=params,
        data_sklearn_compat=False,
        transform_algorithm=None,
        code_init=X,
    )

    dlearn.fit(Y)

    elapsed_time = time.time() - start_time

    dictionary = dlearn.D_
    code = dlearn.X_

    # print("Frobenius norm for X: %s" % np.linalg.norm(code, "fro"))

    sensitivity, specificity, acc = test_fpfn_DL_ocsvm_representations(
        code, y_truth_all, params["clf"]
    )

    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy, sensitivity, specificity, elapsed_time


def Kfold_cv(D, Y_train, y_test, n_nonzero_coefs, params, gamma_ker, gamma,
             update_representations):
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

        n_iterations = 6  # number of DL iterations (K)
        n_features = np.shape(Y_train_folds)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms

        D = np.random.randn(Y_train_folds.shape[1], n_components)
        D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

        start_time = time.time()

        # precompute the Kernel matrix, it s sq root, and sq root's pinv
        # (should be done in kernel_dictionary_learning)
        K = rbf_kernel(Y_train_folds.T, Y_train_folds.T, gamma_ker)
        K += 1e-10 * np.eye(K.shape[0])
        Diag, V = np.linalg.eigh(K)
        Diag_sqr = np.sqrt(Diag)
        # Diag_inv = np.reciprocal(Diag, where=Diag != 0)
        Diag_sqr_inv = np.reciprocal(Diag_sqr, where=Diag_sqr != 0)
        # K_inv = V @ np.diag(Diag_inv) @ V.T
        K_sqr = V @ np.diag(Diag_sqr) @ V.T
        K_sqr_inv = V @ np.diag(Diag_sqr_inv) @ V.T

        params["K"] = K
        params["K_sqr"] = K_sqr
        params["K_sqr_inv"] = K_sqr_inv

        X = dl._dictionary_learning.sparse_encode(
            D.T @ K, D.T @ K @ D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
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

        # Introduce B
        B = np.linalg.lstsq(K, X.T, rcond=None)[0].T
        params["B"] = B

        dlearn = DictionaryLearning(
            n_components=n_components,
            max_iter=n_iterations,
            fit_algorithm=ker_ksvd_supp_DPL_OCSVM,
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

        ba, sensitivity, specificity, test_time = dl_kernel_DPL_predict(
            Y_test_fold,
            Y_train_folds,
            K,
            dictionary,
            y_test_fold,
            params,
            n_nonzero_coefs,
            gamma_ker
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
        "BETA: %s, GAMMA: %s, GAMMA(KER): %s  BA: %s" % (
            params["supp_beta"],
            gamma,
            gamma_ker,
            balanced_accuracy)
    )
    return balanced_accuracy, sensitivity, specificity, train_time, test_time


def ksvd_update_Kfold_cv(beta, filename, nu, Y, y_truth_test,
                         dictionary, gamma_ker, gamma,
                         n_nonzero_coefs, update_representations):
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

    D = dictionary
    D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

    return Kfold_cv(D, Y, y_truth_test, n_nonzero_coefs, params, gamma_ker, gamma,
                    update_representations)


def run_experiment(dir, filename, Y, y_truth_all, beta, nu, cv, gamma_ker,
                   n_nonzero_coefs, update_representation):

    if cv is False:
        Y = Y.T
        Y_norm = np.linalg.norm(Y, ord="fro")
        Y = Y / Y_norm

        # precompute the Kernel matrix, it s sq root, and sq root's pinv
        # (should be done in kernel_dictionary_learning)
        K = rbf_kernel(Y.T, Y.T, gamma_ker)
        K += 1e-10 * np.eye(K.shape[0])
        Diag, V = np.linalg.eigh(K)
        Diag_sqr = np.sqrt(Diag)
        Diag_sqr_inv = np.reciprocal(Diag_sqr, where=Diag_sqr != 0)
        K_sqr = V @ np.diag(Diag_sqr) @ V.T
        K_sqr_inv = V @ np.diag(Diag_sqr_inv) @ V.T

    # for gamma in np.linspace(0.05, 0.5, 10):
    for gamma in np.linspace(0.05, 0.5, 10):
        gamma = round(gamma, 16)

        bas = []
        tnrs = []
        tprs = []

        elapsed_times = []

        dicts = load_dicts(filename, kernel=True)

        for i in range(20):
            if cv is False:
                ba, tpr, tnr, train_time = ksvd_update(
                    beta,
                    gamma,
                    filename,
                    nu[filename],
                    Y,
                    y_truth_all,
                    dicts[i],
                    n_nonzero_coefs,
                    K,
                    K_sqr,
                    K_sqr_inv
                )
                print(
                    "BETA: %s, GAMMA: %s, GAMMA(KER): %s, BA: %s"
                    % (beta, gamma, gamma_ker, ba)
                )
            else:
                ba, tpr, tnr, train_time, test_time = ksvd_update_Kfold_cv(
                    beta,
                    filename,
                    nu[filename],
                    Y,
                    y_truth_all,
                    dicts[i],
                    gamma_ker,
                    gamma,
                    n_nonzero_coefs,
                    update_representation
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
            dir + '/' + f'OCSVM-DPL-kernel_beta{str(beta)}_gamma{str(gamma)}_gammaKer{str(gamma_ker)}.mat',
            {
                'BA': max_ba_beta,
                'meanBA': mean_ba_beta,
                'stdBA': std_ba_beta,
                'TPR': tprs[max_ba_index],
                'TNR': tnrs[max_ba_index],
                'time': elapsed_times[max_ba_index],
                'mean-time': mean_time,
                'dict': dicts[max_ba_index],
            })


def compare_results(dir):
    max_ba = 0
    max_beta_gamma = None
    max_ba_beta_dict = {}
    for path in os.listdir(dir):
        data = scipy.io.loadmat(os.path.join(dir, path))
        name = path.split("OCSVM-DPL-kernel_", 1)[1]
        name = name.split(".mat", 1)[0]
        max_ba_beta_dict[name] = data  # max_ba_beta

    for key in max_ba_beta_dict.keys():
        if max_ba_beta_dict[key]['BA'] > max_ba:
            max_ba = max_ba_beta_dict[key]['BA']
            max_beta_gamma = key

    print("MAX BA: %s (tpr: %s, tnr: %s) mean: %s, std: %s OBTAINED FOR BETA: %s, time: %s(mean: %s)" % (
        max_ba, max_ba_beta_dict[max_beta_gamma]['TPR'],
        max_ba_beta_dict[max_beta_gamma]['TNR'],
        max_ba_beta_dict[max_beta_gamma]['meanBA'],
        max_ba_beta_dict[max_beta_gamma]['stdBA'],
        max_beta_gamma, max_ba_beta_dict[max_beta_gamma]['time'],
        max_ba_beta_dict[max_beta_gamma]['mean-time'])
    )

    beta_tmp = max_beta_gamma.split("beta", 1)[1]
    max_beta = beta_tmp.split("_gamma", 1)[0]
    gamma_tmp = max_beta_gamma.split("_gamma", 1)[1]
    max_gamma = gamma_tmp.split("_gammaKer", 1)[0]
    max_gamma_ker = max_beta_gamma.split("_gammaKer", 1)[1]
    return max_ba_beta_dict[max_beta_gamma]['dict'], max_beta, max_gamma, max_gamma_ker


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

    prediction_update_representation = False

    filename = sys.argv[1]

    if filename not in nu.keys():
        print("Invalid dataset name! You have to specify one of these: "
              + str(list(nu.keys())))
        exit(0)

    gammas = [1/16654, 1/8126, 1/4096, 1/2048, 1/1024, 1/256, 1/128, 1/64, 1/32, 1/12, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
    betas = np.linspace(0.05, 0.5, 10)
    # betas = [0.05, 0.1]

    if cv is False:
        dir = "results/" + filename.split(".mat")[0] + "_DPL_kernel"
    else:
        dir = "results/" + filename.split(".mat")[0] + "_DPL_kernel_KFold"

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

        Parallel(n_jobs=4)(delayed(run_experiment)(
            dir,
            filename,
            y_all,
            y_truth_all,
            round(beta, 16),
            nu, cv, round(gamma, 16),
            n_nonzero_coef[filename],
            prediction_update_representation) for beta in betas for gamma in gammas
        )
    else:
        Y_train, Y_test, y_truth_train, y_truth_test = train_test_split(
            y_all, y_truth_all, test_size=0.2, random_state=37
        )

        Parallel(n_jobs=4)(delayed(run_experiment)(
            dir,
            filename,
            Y_train,
            y_truth_train,
            round(beta, 16),
            nu,
            cv,
            round(gamma, 16),
            n_nonzero_coef[filename],
            prediction_update_representation) for beta in betas for gamma in gammas
        )

    D, max_beta, max_gamma, max_gamma_ker = compare_results(final_directory)
    max_beta = float(max_beta)
    max_gamma = float(max_gamma)
    max_gamma_ker = float(max_gamma_ker)

    if cv is True:
        # Specific for K Fold Validation

        n_iterations = 6  # number of DL iterations (K)
        n_features = np.shape(Y_train.T)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms
        n_nonzero_coefs = n_nonzero_coef[filename]  # sparsity (s)

        # Here we have to generate a new dict because in the kernel formulation the
        # dimension of dictionary A depends on the size of the training data
        D = np.random.rand(np.shape(Y_train.T)[1], n_components)
        D = D @ np.diag(1 / np.linalg.norm(D, axis=0))  # normalize columns of D

        params = {
            "replatoms": 2,
            "atom_norm_tolerance": 1e-10,
            "supp_reg": 1,
            "supp_beta": max_beta,
        }  # NO

        params["nu"] = nu[filename]

        K = rbf_kernel(Y_train, Y_train, max_gamma_ker)
        K += 1e-10 * np.eye(K.shape[0])

        Diag, V = np.linalg.eigh(K)
        Diag_sqr = np.sqrt(Diag)
        Diag_inv = np.reciprocal(Diag, where=Diag != 0)
        Diag_sqr_inv = np.reciprocal(Diag_sqr, where=Diag_sqr != 0)
        K_sqr = V @ np.diag(Diag_sqr) @ V.T
        K_inv = V @ np.diag(Diag_inv) @ V.T
        K_sqr_inv = V @ np.diag(Diag_sqr_inv) @ V.T

        params["K"] = K
        params["K_sqr"] = K_sqr
        params["K_sqr_inv"] = K_sqr_inv

        X = dl._dictionary_learning.sparse_encode(
            D.T @ K, D.T @ K @ D, algorithm="omp", n_nonzero_coefs=n_nonzero_coefs
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

        # Introduce B
        B = np.linalg.lstsq(K, X.T, rcond=None)[0].T
        params["B"] = B

        dlearn = DictionaryLearning(
            n_components=n_components,
            max_iter=n_iterations,
            fit_algorithm=ker_ksvd_supp_DPL_OCSVM,
            n_nonzero_coefs=n_nonzero_coefs,
            dict_init=D,
            params=params,
            data_sklearn_compat=False,
            transform_algorithm=None,
            code_init=X,
        )

        dlearn.fit(Y_train.T)

        ba, sensitivity, specificity, elapsed_time = dl_kernel_DPL_predict(
            Y_test.T,
            Y_train.T,
            K, dlearn.D_,
            y_truth_test,
            params,
            n_nonzero_coefs,
            max_gamma_ker
        )
        print("BA on test set: %s" % str(ba))
