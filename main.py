import numpy as np


def solve(X, A, lamb, display=False, ours=False):
    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    m = A.shape[1]
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    atx = A.T.dot(X)
    inv_a = np.linalg.inv(A.T.dot(A) + np.eye(m))

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
            UVt = 1 * np.dot(U, Vt)
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
        if display and (iter_ == 1 or np.mod(iter_, 50) == 0 or stopC < tol):
            print("iter", iter_, ",mu=", mu, ",rank=",
                  np.linalg.matrix_rank(Z, tol=1e-3*np.linalg.norm(Z, 2)), ",stopALM=", stopC)

        if stopC < tol:
            break
        else:
            Y1 += mu * leq1
            Y2 += mu * leq2
            mu = min(max_mu, mu * rho)

    return Z, E
