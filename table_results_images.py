
import os
import numpy as np

score_type = 'bacc'

for strat in ['S2', 'S3', 'S4']:
    file_name = 'tables_results_images/table_results_{}_score_type_{}.txt'.format(strat, score_type)

    f = open(file_name, 'w')

    f.write(r'\toprule' + '\n')
    f.write('Dataset & oracle & sar-em & lbe & pglin & pusb & threshold' + r'\\' + '\n')
    f.write(r'\midrule' + '\n')

    for ds in ['CIFAR10', 'USPS', 'MNIST', 'Fashion']:
        if score_type == 'bacc':
            index = 6
        elif score_type == 'auc':
            index = 4
        else:
            raise ValueError('{} is not supported as a score type'.format(score_type))
        res = np.zeros((20, 6))

        for i, method in enumerate(['oracle', 'sar-em', 'lbe', 'pglin', 'pusb', 'threshold']):
            res[:, i] = np.loadtxt('results_image/results_method_{}_label_scheme_{}_classifier_nn_ds_{}.txt'.format(method, strat, ds.split('.')[0]))[:, index]

        
        
        table_mean = np.mean(res, axis=0)
        index_max = np.argmax(table_mean[1:]) + 1
        table_std = np.std(res, axis=0)
        table_values = list(map(lambda x, y: '{} \pm {}'.format(x, y), np.round(table_mean, 3), np.round(table_std, 3)))
        table_values[index_max] = '\mathbf{{{}}}'.format(table_values[index_max])
        table_string = [ds.split('.')[0]]
        table_string.extend(list(map(lambda x: '${}$'.format(x), table_values)))
        table_string = ' & '.join(table_string)

        f.write(table_string + r'\\' + '\n')
    
    f.write(r'\bottomrule')
    f.close()

        
        