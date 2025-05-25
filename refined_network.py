import numpy as np
from main import solve
path = 'D:/桌面/LRR/Data/numpy数据网络/'  # 从现有数据中加载


class Multi_Lrr_Model:
    def __init__(self, dat_name):
        self.dat_name = dat_name
        self.data = np.load(path + self.dat_name)
        self.n = self.data.shape[0]
        self.m = np.sum(np.triu(self.data))

    def LRR(self, lam, ours):  # lam = 0.1
        A = np.ones((self.n, self.n)) - np.eye(self.n)
        _, eigvec = np.linalg.eig(self.data)
        Z, E = solve(self.data, A, lam, False, ours=ours)
        temp = Z + Z.T
        Zs = np.sum(abs(temp), axis=0).reshape(self.n, 1)
        new = np.dot(Zs, Zs.T)
        return new


def creat_new_data(data_name, r):
    model = Multi_Lrr_Model(data_name)
    sigular_matrix = model.LRR(0.2, ours=True)
    temp = np.argsort(-sigular_matrix.ravel())  # argsort(x) 返回从小到大的索引值, argsort(-x)返回从大到小的索引值
    temp = [item+1 for item in temp]

    irregular_edge = []
    k = 0
    while len(irregular_edge) / model.m < 2*r:
        item = temp[k]
        # if sigular_matrix.ravel()[item] < 0.0074:
        if item % model.n != 0:
            i = int(np.ceil(item / model.n))
            j = item % model.n
        else:
            i = int(item/model.n)
            j = model.n

        if model.data[i-1, j-1] == 1:
            irregular_edge.append((i-1, j-1))
        k += 1

    new_data = model.data
    for index in irregular_edge:
        new_data[index[0], index[1]] = 0

    np.save('../irregular_datasets/%s(%.2f).npy' % (data_name[:-4], r), new_data)


data_names = ['Bio-CE-GT', 'Celegans', 'Ecoli', 'Facebook', 'Jazz', 'Metabolic', 'PB', 'Soc-wiki-vote']
# data_names = ['T_NW', 'T_WS']
rs = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
for data_name in data_names:
    print(data_name)
    for r in rs:
        creat_new_data(data_name+'.npy', r)



