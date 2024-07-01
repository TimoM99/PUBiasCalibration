# from matplotlib import tight_layout
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from matplotlib.patches import Patch

import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

metric = 'bacc'

if metric == 'bacc':
    start_index = 0
else:
    raise ValueError('The metric {} does not exist.')

fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs = axs.flatten()
for j, strat in enumerate(['S1', 'S2', 'S3', 'S4']):
    axs[j].set_axisbelow(True)
    axs[j].grid(axis='y', alpha=0.5)
    res = np.zeros((20, 2))
    for i, ds in enumerate(os.listdir('data')):
        temp = np.loadtxt('results_threshold/results_threshold_label_scheme_{}_ds_{}.txt'.format(strat, ds))[:, start_index:start_index + 4]
        relative_error = np.maximum(temp[:, 0][:, None] - temp[:, 1:4], np.zeros(temp[:, 1:4].shape))/temp[:, 0][:, None]
        res[i] = np.mean(relative_error[:, 0:2], axis=0)
    
    
    sns.boxplot(res, ax=axs[j], hue_order=[r'$\tau$\texttt{MI}\texttt{calibration}', r'Estimated $\pi$'], showfliers=False, palette=sns.color_palette('husl'))


    axs[j].set_xticks([])
    if j % 2 == 0:
        axs[j].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        axs[j].set_yticklabels(['0', '0.05', '0.10', '0.15', '0.20'], fontsize=20)
    else:
        axs[j].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        axs[j].set_yticklabels([])

    axs[j].set_ylim(0, 0.2)
    axs[j].set_title(strat,
        fontdict={
            'fontsize': 22,
            'alpha': 1,
            'color': 'white'
        },
        bbox={
            'boxstyle':'round',
            'alpha': 1
        },
        pad = 20,
        y=0.80, x=0.06, loc='left')
    

# axs[0].set_ylabel(r'Relative error in Bacc', fontsize=22)
    
legend_elements = [Patch(facecolor=sns.color_palette('husl')[i], edgecolor='black', label=label) for i, label in enumerate([r'$\tau$\texttt{MI}\texttt{calibration}', r'Estimated $\pi$'])]

fig.legend(handles=legend_elements, labels=[r'$\tau$\texttt{MI}\texttt{calibration}', r'Estimated $\pi$'], loc='upper center', bbox_to_anchor=(0.59, 0.11), ncol=2, fontsize=22)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.ylabel('Relative error in Bacc', fontsize=26, labelpad=30)
fig.tight_layout(rect=[0, 0.05, 1, 1])

fig.savefig('boxplot_threshold_comparison.pdf')
        