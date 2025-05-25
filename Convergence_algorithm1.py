import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline


def convergence(X, A, lamb, display=False, ours=False):
    stop_value = []
    iteration = []
    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    m = A.shape[1]
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    atx = A.T.dot(X)
    inv_a = np.linalg.inv(A.T.dot(A) + np.eye(m))
    # Initializing optimization variables
    # intialize
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    E = np.zeros((d, n))  # sparse

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    # Start main loop
    iter_ = 0
    if display:
        print("initial,rank=%f" % np.linalg.matrix_rank(Z))

    while iter_ < maxIter:
        iter_ += 1
        # update J
        if not ours:
            temp = Z + Y2 / mu
            temp = np.nan_to_num(temp)
            U, sigma, Vt = np.linalg.svd(temp)
            V = Vt.T
            svp = len(np.flatnonzero(sigma > 1.0 / mu))
            if svp >= 1:
                sigma = sigma[0:svp] - 1.0 / mu
            else:
                svp = 1
                sigma = np.array([0])
            J = U[:, 0:svp].dot(np.diag(sigma).dot(V[:, 0:svp].T))

        if ours:
            U, sigma, Vt = np.linalg.svd(np.dot(X, J))
            UVt = np.dot(U, Vt)
            J = (mu * Z + Y2 - UVt) / (mu + 2)

        # udpate Z
        Z = inv_a.dot(atx - A.T.dot(E) + J + (A.T.dot(Y1) - Y2) / mu)
        # update E
        xmaz = X - A.dot(Z)
        temp = xmaz + Y1 / mu
        E = np.maximum(0, temp - lamb / mu) + np.minimum(0, temp + lamb / mu)

        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))

        if display and (np.mod(iter_, 10) == 0 or stopC < tol):
            stop_value.append(stopC)
            iteration.append(iter_)
            print("iter", iter_, ",mu=", mu, ",rank=",
                  np.linalg.matrix_rank(Z, tol=1e-3*np.linalg.norm(Z, 2)), ",stopALM=", stopC)

        if stopC < tol:
            break
        else:
            Y1 += mu * leq1
            Y2 += mu * leq2
            mu = min(max_mu, mu * rho)

    return iteration, stop_value


if __name__ == '__main__':
    path = 'D:\\桌面\\LRR\\Data\\numpy数据网络\\'
    fig, ax = plt.subplots(2, 4, figsize=(18, 10))
    for i in range(8):
        data_name = os.listdir(path)[i]
        print(data_name)
        data = np.load(path + data_name)
        n = data.shape[0]
        A = np.ones((n, n)) - np.eye(n)
        iteration, value = convergence(data, A, 0.1, True, ours=True)
        spl = make_interp_spline(iteration, value, 3)
        y1 = spl(iteration)
        if i < 4:
            ax[0, i].scatter(iteration, value, marker='o', color='#7d3f98')
            ax[0, i].plot(iteration, y1, color='indigo')
            ax[0, i].set_title(data_name[:-4])
            ax[0, i].tick_params(direction='in', top=True, right=True, bottom=True, left=True)
            ax[0, i].set_xlabel('Iteration times')
            ax[0, i].set_ylabel(r'$\zeta$')
        elif 4 <= i < 6:
            ax[1, i-4].scatter(iteration, value, marker='o', color='#7d3f98')
            ax[1, i-4].plot(iteration, y1, color='indigo')
            ax[1, i-4].set_title(data_name[:-4])
            ax[1, i-4].tick_params(direction='in', top=True, right=True, bottom=True, left=True)
            ax[1, i-4].set_xlabel('Iteration times')
            ax[1, i-4].set_ylabel(r'$\zeta$')
        else:
            ax[1, i-4].scatter(iteration, value, marker='o', color='#7d3f98')
            ax[1, i-4].plot(iteration, y1, color='indigo')
            ax[1, i-4].set_title(data_name[2:-4])
            ax[1, i-4].tick_params(direction='in', top=True, right=True, bottom=True, left=True)
            ax[1, i-4].set_xlabel('Iteration times')
            ax[1, i-4].set_ylabel(r'$\zeta$')
    plt.savefig("../figures/convergence_algorithm1.png", dpi=600, bbox_inches='tight')
    plt.show()


