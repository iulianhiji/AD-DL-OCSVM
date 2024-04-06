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

# In this code we solve:
#       maximize_x  ||Ax - b||_2^2   s.t.  ||x||_2 = 1
# using CVX and an iterative Power Method (ProxGrad)

import cvxpy as cp
import numpy as np
import numpy.linalg as la
from numpy.linalg import norm as norm


def TRmax_byCVX(A, b):
    # CVX solution using equivalent transformation from:
    # "A. Ben-Tal, M. Teboulle, Hidden convexity in some nonconvex quadratically  constrained quadratic programming"

    [m, n] = np.shape(A)
    g = -A.T @ b
    Q = -A.T @ A
    [d, V] = la.eig(Q)
    d = d.real
    V = V.real
    c = V.T @ g
    c_abs = np.abs(c)
    e = np.ones((n,))

    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.scalar_product(d, y) - 2 * cp.scalar_product(c_abs, cp.sqrt(y))), [y >= 0, e @ y == 1])
    prob.solve(solver=cp.CVXOPT, verbose=False)
    # prob.solve(verbose = False)


    # prob.solve(solver=cp.CVXOPT, verbose=True)

    # print("\nThe optimal value is", prob.value)
    # print("A solution x is")
    # print(y.value)
    ysol = y.value

    z = np.zeros(n)
    for i in range(n):
        z[i] = np.sign(c[i])*np.sqrt(ysol[i])

    x_sol = V @ z

    # print("\n norm ||Ax_sol||: ", norm(A @ x_sol))
    # print("\n Function value CVX: ", norm(A @ x_sol - b), "norm ||x_sol||: ", norm(x_sol))

    return x_sol


def TRmax_byPM(A, b):
    # Power Method (Normalized Proximal Gradient Descent)

    [m, n] = np.shape(A)

    x = np.ones((n, 1))
    x = np.reshape(x / norm(x), (n,))
    L = 2*norm(A.T @ A)
    k = 0
    N = 50

    while k < N:

        res = A @ x - b
        grad_x = 2 * np.transpose(A) @ res
        y = x + (1/L)*grad_x
        x = y / norm(y)

        k = k+1
        # print("\n Function value PM: ", norm(np.dot(A, x) - b)," norm ||x_sol_PM||: ", norm(x))

    x_sol_PM = x

    # print("\n norm ||Ax_sol_PM||: ", norm(A @ x_sol_PM))
    # print("\n !!FINAL !!!Function value PM: ", norm(np.dot(A, x_sol_PM) - b)," norm ||x_sol_PM||: ", norm(x_sol_PM))
    # print("---------------------------------------")

    return x_sol_PM


def TRmax_byPM_2mat(A, K, b):
    # Power Method for the problem
    #       maximize_x  ||AKx - b||_2^2   s.t.  ||x||_2 = 1

    m = np.shape(A)[0]
    p = np.shape(A)[1]
    n = np.shape(K)[1]

    x = np.ones((n, 1))
    x = np.reshape(x / norm(x), (n,))

    v = np.reshape(np.ones((p,)), (p,))
    for i in range(10):
        v = np.dot(A.T, np.dot(A, v))
        v = v / norm(v)
    L = norm(np.dot(A.T, np.dot(A, v)))

    # L = 2*norm(A)**2
    k = 0
    N = 50

    while k < N:

        # rvec = K @ x
        # res = A @ rvec - b
        res = np.dot(A, np.dot(K, x)) - b

        # gvec =  np.transpose(A) @ res
        # grad_x = 2* np.transpose(K) @ gvec
        grad_x = 2 * np.dot(np.transpose(K), np.dot(np.transpose(A), res))

        y = x + (1/L) * grad_x
        x = y / norm(y)
        k = k + 1

    x_sol_PM = x
    # rvec = np.dot(K, x_sol_PM)
    # res_final = np.dot(A, rvec) - b
    # print("\n norm ||Ax_sol_PM||: ", norm(A @ x_sol_PM))
    # print("\n Function value PM: ", norm(res_final)," norm ||x_sol_PM||: ", norm(x_sol_PM))

    return x_sol_PM


if __name__ == "__main__":

    m = 500
    p = 500
    n = 100
    A = np.random.randn(m, p)
    K = np.random.randn(p, n)
    b = np.random.randn(m)

    F = np.matmul(A, K)
    x_sol = TRmax_byCVX(F, b)
    x_sol_PM = TRmax_byPM(F, b)
    print("PM is over!")
    x_sol_PM2 = TRmax_byPM_2mat(A, K, b)
    print("PM2 is over!")
    print("Diference ||CVX-PM||: ", norm(x_sol_PM - x_sol))
    print("Diference ||CVX-PM2||: ", norm(x_sol_PM2 - x_sol))
    # print("\n Sol CVX: ", x_sol)
    # print("\n Sol PM: ", x_sol_PM)
