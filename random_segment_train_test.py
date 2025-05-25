import numpy as np
import random
import copy


def random_get_train_test(A, q):
    n = A.shape[0]
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
