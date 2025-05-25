from model import *
import os
import time


if __name__ == '__main__':
    for i in range(6):
        data_name = os.listdir(path)[i]
        print('data name:', data_name)
        # for r in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        for lam in [0.2]:  # [0.1, 0.2, 0.4, 0.6, 0.8, 1]
            for q in [0.1]:  # 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6
                # print('q:', q, '='*20)
                auc0_list = []
                recall0_list = []
                time0_list = []
                auc1_list = []
                recall1_list = []
                time1_list = []
                for iteration in range(5):
                    time0 = time.time()
                    model = Multi_Lrr_Model(data_name, q)
                    try:
                        A = np.ones((model.n, model.n)) - np.eye(model.n)
                        # X_lrr, Z_lrr, E_lrr = model.LRR(model.train, lam, ours=False)
                        # metrics0 = model.get_metrics(X_lrr, model.test)
                        # auc0_list.append(metrics0[0])
                        # recall0_list.append(metrics0[1])

                        X_ours, Z_ours, E_ours = model.LRR(A, lam, ours=True)
                        metrics1 = model.get_metrics(X_ours, model.test)
                        auc1_list.append(metrics1[0])
                        recall1_list.append(metrics1[1])
                        time1_list.append(time.time() - time0)

                    except np.linalg.LinAlgError:
                        # print('svd不收敛')
                        continue

                # print('lam: %.4f, LRR(mean auc: %.4f, mean recall: %.4f, '
                #       'std auc:%.4f, std recall:%.4f), time consumption: %.4f'
                #       % (lam, np.mean(auc0_list), np.mean(recall0_list),
                #          np.std(auc0_list), np.std(recall0_list), np.mean(time1_list)))

                # print('lam: %.4f, Ours (mean auc: %.4f, mean recall:%.4f, '
                #       'std auc:%.4f, std recall:%.4f), time consumption: %.4f'
                #       % (lam, np.mean(auc1_list), np.mean(recall1_list),
                #          np.std(auc1_list), np.std(recall1_list), np.mean(time1_list)))

                print(np.mean(auc1_list), np.std(auc1_list), np.mean(recall1_list), np.std(recall1_list))

                # print('=' * 20)


