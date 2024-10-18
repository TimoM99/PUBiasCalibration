# from matplotlib import tight_layout
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from matplotlib.patches import Patch

import os

metric = 'bacc'
file_name = 'Library/tables_results_threshold/table_summary_threshold_score_type_{}.txt'.format(metric)

f = open(file_name, 'w')

f.write(r'\toprule' + '\n')
f.write(r'Label strat. & \mitcal & Estimated $\pi$' + r'\\' + '\n')
f.write(r'\midrule' + '\n')

if metric == 'bacc':
    start_index = 0
else:
    raise ValueError('The metric {} does not exist.')

for j, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    res = np.zeros((20, 2))
    for i, ds in enumerate(os.listdir('data/datasets')):
        temp = np.loadtxt('results_threshold/results_threshold_label_scheme_{}_ds_{}.txt'.format(strat, ds))[:, start_index:start_index + 4]
        relative_error = np.maximum(temp[:, 0][:, None] - temp[:, 1:4], np.zeros(temp[:, 1:4].shape))/temp[:, 0][:, None]
        res[i] = np.mean(relative_error[:, 0:2], axis=0)

    f.write('{} & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$'.format(strat, 
                                                    np.round(np.percentile(res[:, 0], 25, axis=0), 3), 
                                                    np.round(np.median(res[:, 0], axis=0),3), 
                                                    np.round(np.percentile(res[:, 0], 75, axis=0),3), 
                                                    np.round(np.percentile(res[:, 1], 25, axis=0),3), 
                                                    np.round(np.median(res[:, 1], axis=0),3), 
                                                    np.round(np.percentile(res[:, 1], 75, axis=0),3))  + r'\\' + '\n')

f.write(r'\bottomrule')
f.close()


    
    


        