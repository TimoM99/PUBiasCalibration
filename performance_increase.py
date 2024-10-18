import numpy as np

    
method_list = ['sar-em', 'pusb', 'pglin', 'lbe', 'PUe']

datasets = ['Abalone', 'Banknote-authentication', 'Breast-w', 'Diabetes', 'Haberman', 'Heart-statlog', 'Ionosphere', 'Isolet', 'Jm1', 'Kc1', 'Madelon', 'Musk', 'Segment', 'Semeion', 'Sonar', 'Spambase', 'Vehicle', 'Waveform-5000', 'Wdbc', 'Yeast'] 
ind = 6 # Index for balanced accuracy according to experiment.py

final_result = np.zeros((4,5))
for m, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    performance_increases = np.zeros((20, 5))
    for i, method in enumerate(method_list):
        for j, ds in enumerate(datasets):
            file_threshold = 'results_UCI/results_parallel_method_' + 'threshold' + '_label_scheme_' + strat + '_classifier_logistic_ds_' + ds + ".txt"    
            res_threshold = np.loadtxt(file_threshold)[:, ind]

            file_comparison = 'results_UCI/results_parallel_method_' + method + '_label_scheme_' + strat + '_classifier_logistic_ds_' + ds + ".txt"    
            res_comparison = np.loadtxt(file_comparison)[:, ind]
            increases = res_threshold - res_comparison
            relative_increases = increases/res_comparison
            avg_relative_increase = np.average(relative_increases)

            performance_increases[j, i] = avg_relative_increase
    
    print(strat)
    for i, method in enumerate(method_list):
        method_increase_strat = np.average(performance_increases[:, i])
        final_result[m, i] = method_increase_strat
        print('Performance increase vs {} is {}'.format(method, method_increase_strat))

for i, method in enumerate(method_list):
    print('Performance increase vs {} is {}'.format(method, np.average(final_result[:, i], axis=0)))

