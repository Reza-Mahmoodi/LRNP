import numpy as np
from main import solve
import matplotlib.pyplot as plt
from random_segment_train_test import random_get_train_test
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from train_test_split import DivideNet
from spurious_network_split import spurious_get_train_test
from random_remove_links import random_remove_train_test

# path = 'D:\\桌面\\LRR\\Data\\numpy数据网络\\'  # 从现有数据中加载
path = '../regular_datasets/WS/'


class Multi_Lrr_Model:
    def __init__(self, dat_name, q):
        self.dat_name = dat_name
        self.q = q
        self.data = np.load(path + self.dat_name)
        self.train, self.test = random_get_train_test(self.data, self.q)
        # self.train, self.test = DivideNet(self.data, self.q)
        # self.train, self.test = random_remove_train_test(self.data, r, q)

        self.n = self.data.shape[0]
        self.L = np.sum(self.test == 1) / 2
        self.Z_list = []
        self.E_list = []

    def LRR(self, A, lam, ours):  # lam = 0.1
        Z, E = solve(self.train, A, lam, False, ours=ours)
        if ours:
            temp = np.dot(A-self.train, Z)  #
            X_hat = temp + temp.T
        else:
            temp = np.dot(A, Z)
            X_hat = temp + temp.T
        return X_hat, Z, E

    def get_metrics(self, X_appro, Test):
        X_appro[X_appro > 0] = 1
        X_appro[X_appro < 1] = 0
        auc = roc_auc_score(Test.ravel(), X_appro.ravel(), average=None)
        recall = recall_score(Test.ravel(), X_appro.ravel(), average=None)[1]
        # precision = precision_score(Test.ravel(), X_appro.ravel(), average=None)[0]
        return auc, recall


def draw(Z, E):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].imshow(Z)
    axes[0].axis('off')
    im1 = axes[1].imshow(E)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.73)
    plt.show()






