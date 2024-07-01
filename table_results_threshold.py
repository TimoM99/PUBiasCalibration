
import os
import numpy as np

score_type = 'bacc'

for strat in ['S1', 'S2', 'S3', 'S4']:
    file_name = 'tables_results_threshold/table_threshold_results_{}_score_type_{}.txt'.format(strat, score_type)

    f = open(file_name, 'w')

    f.write(r'\toprule' + '\n')
    f.write('Dataset & Optimal threshold & MI threshold & Estimated $\pi$ threshold & True $\pi$ threshold' + r'\\' + '\n')
    f.write(r'\midrule' + '\n')

    for i, ds in enumerate(sorted(os.listdir('data'))):
        res = np.loadtxt('results_threshold/results_threshold_label_scheme_{}_ds_{}.txt'.format(strat, ds))

        if score_type == 'tpr':
            start_index = 4
        elif score_type == 'bacc':
            start_index = 0
        elif score_type == 'tnr':
            start_index = 8
        else:
            raise ValueError('{} is not supported as a score type'.format(score_type))
        
        table_mean = np.mean(res[:, start_index:start_index+4], axis=0)
        index_max = np.argmax(table_mean[1:]) + 1
        table_std = np.std(res[:, start_index:start_index+4], axis=0)
        table_values = list(map(lambda x, y: '{} \pm {}'.format(x, y), np.round(table_mean, 3), np.round(table_std, 3)))
        table_values[index_max] = '\mathbf{{{}}}'.format(table_values[index_max])
        table_string = [ds.split('.')[0]]
        table_string.extend(list(map(lambda x: '${}$'.format(x), table_values)))
        table_string = ' & '.join(table_string)

        f.write(table_string + r'\\' + '\n')
    
    f.write(r'\bottomrule')
    f.close()

        
        