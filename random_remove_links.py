import numpy as np
import random
import copy


def random_remove_train_test(A, r, q):
    n = A.shape[0]
    index_nozero = np.argwhere(A != 0)
    R = np.floor(len(index_nozero) * r)
    R = int(R)
    temp = random.sample(list(index_nozero), R)
    for index in temp:
        A[index[0], index[1]] = 0
        A[index[1], index[0]] = 0

    index_nozero = np.argwhere(A != 0)
    L = np.floor(len(index_nozero) * q)
    L = int(L)
    sampled_index = random.sample(list(index_nozero), L)
    train = copy.deepcopy(A)
    for index in sampled_index:
        train[index[0], index[1]] = 0
        train[index[1], index[0]] = 0
    test = np.zeros((n, n))
    for index in sampled_index:
        test[index[0], index[1]] = 1
        test[index[1], index[0]] = 1

    return train, test