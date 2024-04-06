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

import warnings

import numpy as np
from dictlearn.methods._atom import _update_atom, _update_atom_ker
from PGM_routine_git import DoublePGM
from sklearn.svm import OneClassSVM
from TR_routine import TRmax_byPM, TRmax_byPM_2mat

warnings.filterwarnings("ignore", "Solver terminated early.*")
warnings.filterwarnings('ignore', 'Orthogonal matching pursuit.*')


def _ksvd_supp_update_atom_DL_OCSVM(
    F,
    D,
    X,
    atom_index,
    atom_usages,
    params,
):

    ocsvm_magnitude = 1
    if "ocsvm_magnitude" in params:
        ocsvm_magnitude = params["ocsvm_magnitude"]

    A = F.T
    b = -1 * ocsvm_magnitude * params["u"][atom_index] * params["lambda"][atom_usages]

    d = TRmax_byPM(A, b)

    r = F.T @ d + ocsvm_magnitude * params["u"][atom_index] * params["lambda"][atom_usages]
    r_norm = np.linalg.norm(r)

    if params["supp_reg"] == 1:  # L1-regularization
        if r_norm < params["supp_beta"]:
            d = D[:, atom_index]
            x = np.zeros_like(r)
        else:
            x = (1 - params["supp_beta"]/r_norm) * r
    return d, x


def _ksvd_supp_update_atom_DPL_OCSVM(
    F,
    D,
    X,
    atom_index,
    atom_usages,
    params,
):
    Upsilon = params["Y"][:, atom_usages]
    p_start = params["P"][atom_index]
    b = -1 * params["u"][atom_index] * params["lambda"][atom_usages]

    A = F.T
    d, p, _f_val = DoublePGM(A, Upsilon, b, p_start, params["gamma"] * params["alpha"][atom_index], params["supp_beta"], 200)
    params["P"][atom_index] = p

    return d, p @ Upsilon


def _ksvd_supp_update_atom_kernel_DL_OCSVM(
    F,
    K,
    D,
    X,
    atom_index,
    atom_usages,
    params,
):

    A = F.T
    b = -1 * params["u"][atom_index] * params["lambda"][atom_usages]

    K_sqr = params["K_sqr"]
    K_sqr_inv = params["K_sqr_inv"]

    d = TRmax_byPM_2mat(A, K_sqr, b)

    d = K_sqr_inv @ d

    r = F.T @ K @ d + params["u"][atom_index] * params["lambda"][atom_usages]

    r_norm = np.linalg.norm(r)

    if params["supp_reg"] == 1:  # L1-regularization
        if r_norm < params["supp_beta"]:
            d = D[:, atom_index]
            x = np.zeros_like(r)
        else:
            x = (1 - params["supp_beta"]/r_norm) * r
    return d, x


def _ksvd_supp_update_atom_kernel_DPL_OCSVM(
    F,
    K,
    D,
    X,
    atom_index,
    atom_usages,
    params,
):
    K_sqr = params["K_sqr"]
    K_sqr_inv = params["K_sqr_inv"]
    K_i = params["K"][:, atom_usages]

    p_start = params["B"][atom_index]

    A = F.T @ K_sqr
    b = -1 * params["u"][atom_index] * params["lambda"][atom_usages]

    d, p, _f_val = DoublePGM(A, K_i, b, p_start, params["gamma"] * params["alpha"][atom_index], params["supp_beta"], 200)
    d = K_sqr_inv @ d

    params["B"][atom_index] = p

    return d, p @ K_i


def ksvd_supp_DL_OCSVM(Y, D, X, params):
    """
    K-SVD algorithm with coherence reduction
    INPUTS:
        Y -- training signals set
        D -- current dictionary
        X -- sparse representations
    OUTPUTS:
        D -- updated dictionary
        X -- updated representations
    """
    D, X = _update_atom(Y, D, X, params, _ksvd_supp_update_atom_DL_OCSVM)

    # OCSVM step
    clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000, verbose=False).fit(X.T)

    u = clf.coef_[0] / (clf.nu * X.T.shape[0])
    # rho = clf.offset_[0] / (clf.nu * X.T.shape[0])

    params["lambda"] = np.zeros(X.T.shape[0])
    params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

    params["u"] = u
    params["clf"] = clf

    return D, X


def ksvd_supp_DPL_OCSVM(Y, D, X, params):
    """
    K-SVD algorithm with coherence reduction
    INPUTS:
        Y -- training signals set
        D -- current dictionary
        X -- sparse representations
    OUTPUTS:
        D -- updated dictionary
    """

    D, X = _update_atom(Y, D, X, params, _ksvd_supp_update_atom_DPL_OCSVM)

    # OCSVM step
    clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)
    u = clf.coef_[0] / (clf.nu * X.T.shape[0])

    params["lambda"] = np.zeros(X.T.shape[0])
    params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

    params["u"] = u
    params["clf"] = clf

    return D, X


def ker_ksvd_supp_DPL_OCSVM(K, A, X, params):
    """
    K-SVD algorithm with coherence reduction
    INPUTS:
        K : ndarray of shape (n_samples, n_samples)
        Kernel matrix.

        A : ndarray of shape (n_samples, n_components)
            Initial dictionary, with normalized columns.

        X : ndarray of shape (n_components, n_features)
            The sparse codes.
        OUTPUTS:
        D -- updated dictionary
    """
    A, X = _update_atom_ker(K, A, X, params, _ksvd_supp_update_atom_kernel_DPL_OCSVM)

    # OCSVM step
    clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)

    u = clf.coef_[0] / (clf.nu * X.T.shape[0])

    params["lambda"] = np.zeros(X.T.shape[0])
    params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

    params["u"] = u
    params["clf"] = clf

    return A, X


def ker_ksvd_supp_DL_OCSVM(K, A, X, params):
    """
    K-SVD algorithm with coherence reduction
    INPUTS:
        K : ndarray of shape (n_samples, n_samples)
        Kernel matrix.

        A : ndarray of shape (n_samples, n_components)
            Initial dictionary, with normalized columns.

        X : ndarray of shape (n_components, n_features)
            The sparse codes.
        OUTPUTS:
        D -- updated dictionary
    """
    A, X = _update_atom_ker(K, A, X, params, _ksvd_supp_update_atom_kernel_DL_OCSVM)

    # OCSVM step
    clf = OneClassSVM(kernel="linear", nu=params["nu"], max_iter=5000).fit(X.T)

    u = clf.coef_[0] / (clf.nu * X.T.shape[0])

    params["lambda"] = np.zeros(X.T.shape[0])
    params["lambda"][clf.support_] = clf.dual_coef_ / (clf.nu * X.T.shape[0])

    # update u
    params["u"] = u
    params["clf"] = clf

    return A, X

