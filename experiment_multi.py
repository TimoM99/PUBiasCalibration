from functools import partial
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import src.helper_files.km as km
import pandas as pd
import multiprocessing
import mkl

import src.Models.PUSB as pusb
from src.Models.PUSB import PUSB
import src.Models.LBE as lbe
from src.Models.LBE import LBE
import src.Models.PGlin as pgl
from src.Models.PGlin import PUGerych
import src.Models.basic as basic
from src.Models.basic import PUbasic
import src.Models.SAREM as sarem
from src.Models.SAREM import SAREM
import src.Models.threshold as threshold
from src.Models.threshold import PUthreshold
from src.helper_files.utils import make_binary_class, sigmoid 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_curve, auc, precision_recall_curve
import warnings




def experiment_lr(ds, nsym, strat):
    label_strat = strat

    # Load data:
    df_name = 'data/' + ds + '.csv'
    df = pd.read_csv(df_name, sep=',')
    del df['BinClass']
    df = df.to_numpy()
    p = df.shape[1]-1
    Xall = df[:,0:p]
    yall = df[:,p]
    yall = make_binary_class(yall)
    
    
    for method in ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle']:
        print('\n Method:', method)
        
        results = np.zeros((nsym, 8))
        
        for sym in np.arange(0, nsym, 1):
            np.random.seed(sym)
            km.seed(sym)

            pusb.seed(sym)
            lbe.seed(sym)
            pgl.seed(sym)
            sarem.seed(sym)
            threshold.seed(sym)
            basic.seed(sym)
            
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
        

            start_time = time.time()
            if method == 'oracle':
                model = PUbasic()
                model.fit(X,y)
            else:
                if method == 'threshold':
                    model = PUthreshold() 
                elif method == 'sar-em':
                    model = SAREM()
                elif method == 'lbe':
                    model = LBE()
                elif method == 'pglin':
                    model = PUGerych()
                elif method == 'pusb':
                    X_mixture = X[np.where(s==0)[0],:]
                    X_component = X[np.where(s==1)[0],:]
                    km1 = km.km_default(X_mixture, X_component)
                    est_pi = (1-np.mean(s))*km1[1] +  np.mean(s)
                    model = PUSB(est_pi, Xtest, ytest)
                model.fit(X, s)
            end_time = time.time()
            run_time = end_time - start_time
            
            prob_y_test = model.predict_proba(Xtest)[:, 1]
            if np.any(np.isnan(prob_y_test)):
                prob_y_test[np.where(np.isnan(prob_y_test))]= np.mean(s)
        
            acc = accuracy_score(ytest, np.where(prob_y_test>0.5,1,0))
            f1 = f1_score(ytest, np.where(prob_y_test>0.5,1,0))
            tpr = np.count_nonzero(np.where(prob_y_test>0.5,1,0)[ytest == 1] == 1)/np.count_nonzero(ytest == 1)
            tnr = np.count_nonzero(np.where(prob_y_test>0.5,1,0)[ytest == 0] == 0)/np.count_nonzero(ytest == 0)
            fpr_thr, tpr_thr, thr = roc_curve(ytest, prob_y_test, pos_label=1)
            roc_auc = auc(fpr_thr, tpr_thr)
            prec_thr, recall_thr, thr = precision_recall_curve(ytest, prob_y_test)
            pr_auc = auc(recall_thr,prec_thr)
            bacc = balanced_accuracy_score(ytest, np.where(prob_y_test>0.5,1,0))
            
            results[sym,0] = acc
            results[sym,1] = f1
            results[sym,2] = tpr
            results[sym,3] = tnr
            results[sym,4] = roc_auc
            results[sym,5] = pr_auc
            results[sym,6] = bacc
            results[sym,7] = run_time
            
            file_out = 'results_UCI/results_method_' + method + '_label_scheme_'+ label_strat + '_classifier_' + 'logistic' + '_ds_' + ds + ".txt"  
            np.savetxt(file_out, results)

    
if __name__ == "__main__":
    # Lock every process to one thread to get accurate training times.
    mkl.set_num_threads(1)
    warnings.filterwarnings("ignore")
    for strat in ['S1', 'S2', 'S3', 'S4']:
        datasets = ['Breast-w','Diabetes','Kc1','Spambase','Wdbc','Banknote-authentication',
                    'Heart-statlog','Jm1','Ionosphere','Sonar','Haberman','Segment','Waveform-5000','Yeast','Musk','Abalone','Isolet','Madelon','Semeion','Vehicle']
        pool = multiprocessing.Pool(20)
        pool.map(partial(experiment_lr, nsym=20, strat=strat), datasets)
        pool.close()
