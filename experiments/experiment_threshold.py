"""
This script runs the experiments for Q4.
"""


import time
import pandas as pd
import numpy as np

from PUBiasCalibration.helper_files.utils import make_binary_class, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import PUBiasCalibration.threshold_optimizer as threshold_optimizer
from PUBiasCalibration.threshold_optimizer import ThresholdOptimizer
from tqdm import tqdm
import PUBiasCalibration.helper_files.km as km
from PUBiasCalibration.helper_files.km import km_default
from multiprocessing import Pool

# We try to find the optimal threshold that maximizes the balanced accuracy
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

    for label_strat in ['S4']:

        results = []
        for sym in tqdm(np.arange(0, nsym, 1)):
            # Set seeds
            np.random.seed(sym)
            km.seed(sym)
            threshold_optimizer.seed(sym)
            

            X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=0.25, random_state=sym)
            n = X.shape[0]

            # Make PU data set
            prob_true = LogisticRegression(random_state=sym).fit(X, y).predict_proba(X)[:, 1]
            prob_true[np.where(prob_true==1)] = 0.999
            prob_true[np.where(prob_true==0)] = 0.001
            s = np.zeros(n)
            if label_strat == 'S1':
                prop_score = np.full(n, 0.1)
            elif label_strat == 'S2':
                prop_score = 0.1 * prob_true
            elif label_strat == 'S3':
                prop_score = sigmoid(-0.5 * prob_true - 1.5)
            elif label_strat == 'S4':
                lin_pred = np.log(prob_true/(1 - prob_true))
                prop_score = 0.5 * sigmoid(-0.5 * lin_pred)

            while np.sum(s) == 0:
                for i in np.arange(0,n,1):
                    if y[i]==1:
                        s[i] = np.random.binomial(1, prop_score[i], size=1)
            
            ntc = LogisticRegression(max_iter=1000, random_state=sym)

            ntc.fit(X, s)

            sx = ntc.predict_proba(X)[:,1]
        
            sx[np.where(sx==1)] = 0.999
            sx[np.where(sx==0)] = 0.001

            lin_pred = np.log(sx/(1-sx))

            start_time = time.time()
            optimal_t = find_optimal_threshold(lin_pred, y)
            optimal_time = time.time() - start_time
            
            start_time = time.time()
            MI_t_100 = ThresholdOptimizer(3, 100).find_threshold(lin_pred[s==0])
            N100_time = time.time() - start_time
            
            start_time = time.time()
            X_mixture = X[np.where(s==0)[0],:]
            X_component = X[np.where(s==1)[0],:]
            km1 = km_default(X_mixture, X_component)
            est_pi = (1-np.mean(s))*km1[1] +  np.mean(s)
            pi_time = time.time() - start_time
            
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
            

            scores.append(est_pi)
            scores.append(true_pi)
            scores.extend([optimal_time, N100_time, pi_time])
            results.append(scores)
        
            file_out = 'results_threshold/results_threshold_label_scheme_'+ label_strat + '_ds_' + ds + ".txt"  
            np.savetxt(file_out, results, fmt='%s')


if __name__ == '__main__':
    datasets = ['Breast-w','Diabetes','Kc1','Spambase','Wdbc','Banknote-authentication',
                'Heart-statlog','Jm1','Ionosphere','Sonar','Haberman','Segment','Waveform-5000','Yeast','Musk','Abalone','Isolet','Madelon','Semeion','Vehicle']
    pool = Pool(processes=20)
    pool.map(threshold_experiment, datasets)
    pool.close()













    