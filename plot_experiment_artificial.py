from operator import index
import os

# os.chdir('/Users/timo/Documents/Work/BiasCalibration/Library')
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


#Set parameters:
# measure = "bacc"
# ds = 'MNIST'
# c = 0.3
nsym = 20
classifier = "lr"
method_list = ['sar-em','lbe','pglin', 'pusb', 'threshold', 'oracle']
distr_list = ['norm','unif','lognorm']

method_list_mod = method_list.copy()    
method_list_mod[method_list_mod.index('oracle')]=r'\textsc{Oracle}'
# method_list_mod[method_list_mod.index('naive')]='NAIVE'
method_list_mod[method_list_mod.index('sar-em')]=r'\textsc{SAR-EM}'
method_list_mod[method_list_mod.index('lbe')]=r'\textsc{LBE}'
method_list_mod[method_list_mod.index('pglin')]=r'\textsc{PGlin}'
method_list_mod[method_list_mod.index('threshold')]=r'\texttt{NTC-}$\tau$\texttt{MI}'
method_list_mod[method_list_mod.index('pusb')]=r'\textsc{PUSB}'

distr_list_mod = distr_list.copy()    
distr_list_mod[distr_list_mod.index('norm')]='Normal'
distr_list_mod[distr_list_mod.index('unif')]='Uniform'
distr_list_mod[distr_list_mod.index('lognorm')]='Lognormal'


index_metric = 6
for label_strat in ['S1', 'S2', 'S3', 'S4']:
    df = np.zeros((nsym,len(method_list), 3))
    for j, dist in enumerate(distr_list):

       
        for i, method in enumerate(method_list):
            file_out = 'Library/results_artificial/results_method_' + method + '_label_scheme_' + label_strat + '_xdistr_' + dist + ".txt"    
            res = np.loadtxt(file_out)
            df[:,i, j]= res[:,index_metric] 
                
    name_fig = 'Library/plots_results_artificial/plot_label_strat' + label_strat + '.pdf'
            
            
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
    for j in range(3):
        performance_distr = df[:, :, j]
        sns.set_style("whitegrid")
        sns.boxplot(ax=axs[j], data=performance_distr, orient='h')
        if j == 1:
            axs[j].set_xlabel('Balanced accuracy', fontsize=26)
        axs[j].set_xticks([0.6, 0.7, 0.8])
        axs[j].set_xticklabels([0.6, 0.7, 0.8], fontsize=26)
        axs[j].set_yticks([0, 1, 2, 3, 4, 5])
        axs[j].grid(axis='x', alpha=0.5)
        axs[j].set_yticklabels(method_list_mod, fontsize=22)
        if j in [1, 2]:
            axs[j].set_yticklabels([])
        # axs[j].set_title('Distribution: ' + distr_list[j])
        # Add blue box to top right
        axs[j].text(0.97, 0.95, distr_list_mod[j], fontsize=26, color='white', bbox=dict(boxstyle='round,pad=0.3'), horizontalalignment='right', verticalalignment='top', transform=axs[j].transAxes)
        axs[j].set_xlim(0.5, 0.9)
    plt.tight_layout()
    plt.savefig(name_fig, format="pdf", bbox_inches="tight")
    # measures_mean = np.apply_along_axis(np.mean,0,df1)
    # measures_std = np.apply_along_axis(np.std,0,df1)
    # print(method_list,'\n',measures_mean)
    # print(method_list,'\n',measures_std)

    
    # sns.set_style("whitegrid")
    # sns.set(font_scale=2)
    # f, ax = plt.subplots()
    # #plot=sns.violinplot(df1,orient="h",inner="points") 
    # plot=sns.boxplot(df1,orient="h")
    # plt.xlabel(measure1)
    # plt.xlim(0.5,1.0)
    # plt.title('Labelling strategy: ' + 'SCAR, c = ' + str(c))
    # plt.savefig(name_fig, format="pdf", bbox_inches="tight")
