import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

method1 = 'pusb' #x-axis
method2 = 'threshold' #y-axis
score_type = 'bacc'

fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs = axs.flatten()
for j, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    res = np.zeros((20, 2))
    for i, ds in enumerate(sorted(os.listdir('data/datasets'))):
        if score_type == 'bacc':
            index = 6
        elif score_type == 'auc':
            index = 4
        elif score_type == 'acc':
            index = 0
        else:
            raise ValueError('{} is not supported as a score type'.format(score_type))
        
        
        res[i, 0] = np.mean(np.loadtxt('results_UCI/results_used_in_paper_AI_stats/results_parallel_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format(method1, strat, ds.split('.')[0]))[:, index])
        res[i, 1] = np.mean(np.loadtxt('results_UCI/results_used_in_paper_AI_stats/results_parallel_method_{}_label_scheme_{}_classifier_logistic_ds_{}.txt'.format(method2, strat, ds.split('.')[0]))[:, index])

    w = res[:, 1] > (res[:, 0] + 0.01)
    l = res[:, 1] < (res[:, 0] - 0.01)
    d = ~w & ~l
    axs[j].scatter(res[:,0][w], res[:,1][w], marker='+', s=250, label='Win', c='g')
    axs[j].scatter(res[:,0][l], res[:,1][l], marker='_', s=250, label='Loss', c='r')
    axs[j].scatter(res[:,0][d], res[:,1][d], marker='.', s=250, label='Draw', c='grey')
    axs[j].plot([0, 1], [0, 1], transform=axs[j].transAxes, linestyle='dashed', alpha=0.3, color='grey')
    axs[j].set_xticks([0.6, 0.8, 1.0])
    axs[j].set_xticklabels([0.6, 0.8, 1.0], fontsize=22)
    axs[j].set_yticks([0.6, 0.8, 1.0])
    axs[j].set_yticklabels([0.6, 0.8, 1.0], fontsize=22)
    if j in [0, 2]:
        axs[j].set_ylabel(r'Balanced Accuracy - \texttt{NTC-}$\tau$\texttt{MI}', fontsize=25)
    if j in [1, 3]:
        # axs[j].set_yticks([0.6, 0.8, 1.0])
        axs[j].set_yticklabels([])
    if j in [0, 1]:
        axs[j].set_xticklabels([])
    if j in [2, 3]:
        axs[j].set_xlabel(r'Balanced Accuracy - $\textsc{PUSB}$', fontsize=25)
    axs[j].set_xlim(0.5, 1)
    axs[j].set_ylim(0.5, 1)
    axs[j].set_title(strat,
        fontdict={
            'fontsize': 24,
            'alpha': 1,
            'color': 'white'
        },
        bbox={
            'boxstyle':'round',
            'alpha': 1
        },
        pad = 20,
        y=0.81, x=0.06, loc='left')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=22)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to create space for the legend
fig.savefig('plots_results_UCI/plot_{}_vs_{}.pdf'.format(method1, method2))
