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

import numpy as np
from numpy.linalg import norm as norm

# This code solves:
# min_x  min_p   1/2 ||xp^TY - R||^2 + beta*||Y^Tp||_2 + alpha ||Y^Tp||_1
#
# Notation: A:=R^T, x:=d
# The problem is formulated as in the proof of Prop. 5
# The algorithm is the prima-dual gradient method for smooth min-max optimization
# The penalty parameter were chosen equal to 1 (to be tuned)
# The number of iterations is to be tuned


def DoublePGM(A, Y, b, p_start, alpha, beta, N):
    # Projected Gradient Descent

    [m, n] = np.shape(A)
    [d, ns] = np.shape(Y)

    x = np.ones((n, 1))
    x = np.reshape(x / norm(x), (n,))
    # p = np.random.randn(d)
    p = p_start
    prodYp = np.dot(np.transpose(Y), p)
    fval = 0.5 * norm(np.transpose(A) - np.outer(x, np.transpose(prodYp)), 'fro') ** 2 + beta * norm(
        prodYp) + alpha * norm(prodYp, 1)

    L_d = norm(A.T @ A, 'fro')
    L_p = norm(Y.T @ Y, 'fro')
    k = 0

    while k < N:
        ress = A @ x - b
        prodYp = np.dot(np.transpose(Y), p)

        # Min step in tau1-tau2
        tau1 = np.sign(prodYp)
        nrm_prodYp = norm(prodYp)
        tau2 = prodYp / nrm_prodYp

        # Gradient step in p
        v = prodYp - ress + (beta * tau2) + (alpha * tau1)
        grad_p = np.dot(Y, v)
        p = p - (1 / L_p) * grad_p

        # Gradient step in d
        grad_x = np.matmul(np.transpose(A), prodYp)
        y = x + (1 / L_d) * grad_x
        x = y / norm(y)

        fval = 0.5 * norm(np.transpose(A) - np.outer(x, np.transpose(prodYp)), 'fro') ** 2 + beta * norm(
            prodYp) + alpha * norm(prodYp, 1)
        # print("\n Function value PM: ", fval, k)
        # if k < 3: print(x)

        k = k + 1

    x_sol_PM = x
    p_sol = p

    return x_sol_PM, p_sol, fval


"""p = 500
n = 100
alpha = 0.01
beta = 0.05
A = 5 * np.random.randn(p, n)
Y = 5 * np.random.randn(n, p)
b = np.random.randn(p)

p = np.random.randn(n)

A = A / np.linalg.norm(A, "fro")
Y = Y / np.linalg.norm(Y, "fro")

[x_sol, p_sol, fval] = DoublePGM(A, Y, b, p, alpha, beta, 200)"""