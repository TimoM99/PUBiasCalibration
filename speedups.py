import os
import numpy as np

# results = np.zeros((4,4))
# stds = np.zeros((4,4))
res = np.zeros((4, 20, 4))
for k, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    for j, ds in enumerate(os.listdir('data')):
        avg_time_proposed = np.mean(np.loadtxt('results_UCI/results_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format('threshold', strat, ds.split('.')[0]))[:, 7])
        for i, method in enumerate(['sar-em', 'lbe', 'pglin', 'pusb']):
            res[k, j, i] = np.mean(np.loadtxt('results_UCI/results_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format(method, strat, ds.split('.')[0]))[:,7])/avg_time_proposed


print(np.mean(res, axis=0))
print(np.std(res, axis=0))