
import numpy as np

# results = np.zeros((4,4))
# stds = np.zeros((4,4))
file_name = 'tables_results_UCI/table_times.txt'
datasets = ['Abalone', 'Banknote-authentication', 'Breast-w', 'Diabetes', 'Haberman', 'Heart-statlog', 'Ionosphere', 'Isolet', 'Jm1', 'Kc1', 'Madelon', 'Musk', 'Segment', 'Semeion', 'Sonar', 'Spambase', 'Vehicle', 'Waveform-5000', 'Wdbc', 'Yeast'] 
f = open(file_name, 'w')

f.write(r'\toprule' + '\n')
f.write('Dataset & oracle & sar-em & lbe & pglin & pusb & threshold' + r'\\' + '\n')
f.write(r'\midrule' + '\n')

res = np.zeros((4, 20, 6))
for k, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    for j, ds in enumerate(datasets):
        # avg_time_proposed = np.mean(np.loadtxt('results_UCI/results_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format('threshold', strat, ds.split('.')[0]))[:, 7])
        for i, method in enumerate(['sar-em', 'PUe', 'lbe', 'pglin', 'pusb', 'threshold']):
            res[k, j, i] = np.mean(np.loadtxt('results_UCI/results_parallel_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format(method, strat, ds.split('.')[0]))[:,7])

table_mean = np.mean(res, axis=0)
table_std = np.std(res, axis=0)

table_median = np.median(res, axis=0)
table_25 = np.percentile(res, 25, axis=0)
table_75 = np.percentile(res, 75, axis=0)
for j, ds in enumerate(datasets):
    # table_values = list(map(lambda x, y: '{:.2f} & {:.2f}'.format(x, y), np.round(table_mean[j], 2), np.round(table_std[j], 2)))
    table_values = list(map(lambda x, y, z: '\\textcolor{{gray}}{{{:.2f}}} & {{{:.2f}}} & \\textcolor{{gray}}{{{:.2f}}}'.format(x, y, z), np.round(table_25[j], 2), np.round(table_median[j], 2), np.round(table_75[j], 2)))
    table_string = [ds.split('.')[0]]
    table_string.extend(list(map(lambda x: '{}'.format(x), table_values)))
    table_string = ' & '.join(table_string)

    f.write(table_string + r'\\' + '\n')
    
f.close()

        
        