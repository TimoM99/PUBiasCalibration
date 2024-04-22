import os
import time
import pandas as pd
import numpy as np
import seaborn as sns

from utils import make_binary_class, sigmoid
from sklearn.model_selection import train_test_split
from Models.basic import PUbasic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score
from threshold_optimizer import ThresholdOptimizer
from tqdm import tqdm
from km import km_default
from multiprocessing import Pool
from matplotlib import axis, pyplot as plt

def find_optimal_threshold(Z, labels):
    possible_t = np.unique(Z)
    best_t = None
    best_bacc = 0
    for i, t in enumerate(possible_t):
        pred = np.where(Z > t, 1, 0)
        bacc = balanced_accuracy_score(labels, pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = (t + possible_t[i + 1])/2
    return best_t

nsym = 20

def threshold_experiment(ds):
    ds_name = ds
    print(ds_name)

    df = pd.read_csv('data/'+ds+'.csv', sep=',')
    del df['BinClass']
    df = df.to_numpy()
    p = df.shape[1]-1
    Xall = df[:,0:p]
    yall = df[:,p]
    yall = make_binary_class(yall)

    # for label_strat in ['S1', 'S2', 'S3', 'S4']:
    for label_strat in ['S4']:

        results = []
        # Results order ['Optimal bacc', 'MI bacc n=100', 'MI bacc n=1000', 'MI bacc n=10000', 'Est pi bacc', 'True pi bacc', 'Est pi', 'true pi', 'optimal_time', 'N100_time', 'N1000_time', 'N1000_time', 'pi_time']
        for sym in tqdm(np.arange(0, nsym, 1)):
            X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=0.25, random_state=sym)

            n = X.shape[0]

            prob_true = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
            prob_true[np.where(prob_true==1)] = 0.999
            prob_true[np.where(prob_true==0)] = 0.001
            # Make PU data set
            s = np.zeros(n)
            if label_strat == 'S1':
                prop_score = np.full(n, 0.1)
            elif label_strat == 'S2':
                prop_score = 0.1 * prob_true
            elif label_strat == 'S3':
                # prob_true = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
                prop_score = sigmoid(-0.5 * prob_true - 1.5)
            elif label_strat == 'S4':
                # prob_true = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
                lin_pred = np.log(prob_true/(1 - prob_true))
                prop_score = 0.5 * sigmoid(-0.5 * lin_pred)

            while np.sum(s) == 0:
                for i in np.arange(0,n,1):
                    if y[i]==1:
                        s[i] = np.random.binomial(1, prop_score[i], size=1)
            
            ntc = LogisticRegression(max_iter=1000)

            ntc.fit(X, s)

            sx = ntc.predict_proba(X)[:,1]
        
            sx[np.where(sx==1)] = 0.999
            sx[np.where(sx==0)] = 0.001

            lin_pred = np.log(sx/(1-sx))

            start_time = time.time()
            optimal_t = find_optimal_threshold(lin_pred, y)
            optimal_time = time.time() - start_time
            # print('Finding the optimal threshold took {} seconds'.format(end_time - start_time))
            start_time = time.time()
            MI_t_100 = ThresholdOptimizer(3, 100).find_threshold(lin_pred[s==0])
            N100_time = time.time() - start_time
            # print('Finding the n=100 MI threshold took {} seconds'.format(end_time - start_time))
            # start_time = time.time()
            # MI_t_1000 = ThresholdOptimizer(3, 1000).find_threshold(lin_pred)
            # N1000_time = time.time() - start_time
            # print('Finding the n=1000 MI threshold took {} seconds'.format(end_time - start_time))
            # start_time = time.time()
            # MI_t_10000 = ThresholdOptimizer(3, 10000).find_threshold(lin_pred)
            # N10000_time = time.time() - start_time
            # print('Finding the n=10000 MI threshold took {} seconds'.format(end_time - start_time))

            start_time = time.time()
            X_mixture = X[np.where(s==0)[0],:]
            X_component = X[np.where(s==1)[0],:]
            km1 = km_default(X_mixture, X_component)
            est_pi = (1-np.mean(s))*km1[1] +  np.mean(s)
            pi_time = time.time() - start_time
            # print('Finding the estimated pi took {} seconds'.format(end_time - start_time))
            true_pi = np.sum(y)/len(y)

            est_pi_t = np.sort(lin_pred)[int(n*(1-est_pi)) - 1] + (n*(1-est_pi) % 1) * (np.sort(lin_pred)[int(n*(1-est_pi))] - np.sort(lin_pred)[int(n*(1-est_pi)) - 1])
            true_pi_t = np.sort(lin_pred)[int(n*(1-true_pi)) - 1] + (n*(1-true_pi) % 1) * (np.sort(lin_pred)[int(n*(1-true_pi))] - np.sort(lin_pred)[int(n*(1-true_pi)) - 1])
        
            thresholds = [optimal_t, MI_t_100, est_pi_t, true_pi_t]
            scores = []
            test_lin_pred = ntc.decision_function(Xtest)
            for t in thresholds:
                test_pred = np.where(test_lin_pred > t, 1, 0)
                scores.append(balanced_accuracy_score(ytest, test_pred))

            for t in thresholds:
                test_pred = np.where(test_lin_pred > t, 1, 0)
                tp = np.count_nonzero(test_pred[ytest == 1] == 1)
                p = np.count_nonzero(ytest == 1)
                scores.append(tp/p)

            for t in thresholds:
                test_pred = np.where(test_lin_pred > t, 1, 0)
                tn = np.count_nonzero(test_pred[ytest == 0] == 0)
                n = np.count_nonzero(ytest == 0)
                scores.append(tn/n)
            

            # sns.kdeplot(lin_pred[(y==0)], label='Negatives', fill=True)
            # sns.kdeplot(lin_pred[(y==1)], label='Positives', fill=True)
            # sns.kdeplot(lin_pred[(y==0) & (s==0)], label='Negative unlabeled', fill=True)
            # sns.kdeplot(lin_pred[(y==1) & (s==0)], label='Positive unlabeled', fill=True)
            # # sns.kdeplot(lin_pred[s==0], label='Negatives', fill=True)
            # plt.axvline(x=est_pi_t, color='r')
            # plt.axvline(x=true_pi_t, color='g')
            # plt.axvline(x=MI_t_100, c='b')
            # plt.axvline(x=optimal_t, c='y')
            # plt.legend()
            # plt.show()
            # # print(est_pi)
            # # print(true_pi)
            # # print(scores)

            scores.append(est_pi)
            scores.append(true_pi)
            scores.extend([optimal_time, N100_time, pi_time])
            results.append(scores)
        
            file_out = 'results_threshold/results_threshold_label_scheme_'+ label_strat + '_ds_' + ds + ".txt"  
            np.savetxt(file_out, results, fmt='%s')


if __name__ == '__main__':
    # threshold_experiment('Abalone.csv')
    datasets = ['Breast-w','Diabetes','Kc1','Spambase','Wdbc','Banknote-authentication',
                'Heart-statlog','Jm1','Ionosphere','Sonar','Haberman','Segment','Waveform-5000','Yeast','Musk','Abalone','Isolet','Madelon','Semeion','Vehicle']
    pool = Pool()
    pool.map(threshold_experiment, datasets)
    pool.close()













    