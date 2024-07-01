import argparse

import numpy as np



def main(args):
    score_folder = str(args.folder)
    label_strat = str(args.strat)
    classifier = str(args.clf)
    
    method_list = ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle']

    datasets = ['Abalone', 'Banknote-authentication', 'Breast-w', 'Diabetes', 'Haberman', 'Heart-statlog', 'Ionosphere', 'Isolet', 'Jm1', 'Kc1', 'Madelon', 'Musk', 'Segment', 'Semeion', 'Sonar', 'Spambase', 'Vehicle', 'Waveform-5000', 'Wdbc', 'Yeast'] 
    ind = 1 # Index for balanced accuracy according to experiment.py
    # datasets = ['CIFAR10', 'USPS', 'MNIST', 'Fashion']
    

    for method in method_list:
        wins = 0
        losses = 0
        for ds in datasets:
            file_threshold = score_folder + '/results_method_' + 'threshold' + '_label_scheme_' + label_strat + '_classifier_' + classifier + '_ds_' + ds + ".txt"    
            res_threshold = np.loadtxt(file_threshold)
            avg_bacc_threshold = np.average(res_threshold[:, ind])

            file_comparison = score_folder + '/results_method_' + method + '_label_scheme_' + label_strat + '_classifier_' + classifier + '_ds_' + ds + ".txt"    
            res_comparison = np.loadtxt(file_comparison)
            avg_bacc_comparison = np.average(res_comparison[:, ind])

            if avg_bacc_threshold > avg_bacc_comparison + 0.01:
                wins += 1
            elif avg_bacc_comparison > avg_bacc_threshold + 0.01:
                losses += 1
        
        print('Wins/Losses vs {}:'.format(method))
        print('{}/{}'.format(wins, losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', required=True)
    parser.add_argument('-strat', required=True)
    parser.add_argument('-clf', required=True)


    args = parser.parse_args()
    main(args)