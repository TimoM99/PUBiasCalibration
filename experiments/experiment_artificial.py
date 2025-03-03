# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:57:46 2023

@author: teiss

This script runs the experiments on artificial datasets to answer Q3.
"""

import random
from PUBiasCalibration.helper_files import km
# import PUBiasCalibration.helper_files.km as km

from PUBiasCalibration.Models import PUSB as pusb
from PUBiasCalibration.Models.PUSB import PUSB
import PUBiasCalibration.Models.LBE as lbe
from PUBiasCalibration.Models.LBE import LBE
import PUBiasCalibration.Models.PGlin as pgl
from PUBiasCalibration.Models.PGlin import PUGerych
import PUBiasCalibration.Models.basic as basic
from PUBiasCalibration.Models.basic import PUbasic
import PUBiasCalibration.Models.SAREM as sarem
from PUBiasCalibration.Models.SAREM import SAREM
import PUBiasCalibration.Models.threshold as threshold
from PUBiasCalibration.Models.PUe import PUe
import PUBiasCalibration.Models.PUe as pue
from PUBiasCalibration.Models.threshold import PUthreshold
from artificial import generate_artificial_data_gen

import os
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc, roc_curve, precision_recall_curve
from PUBiasCalibration.helper_files.utils import sigmoid

#Set parameters:
p=10 # number of features
n=2000 # number of samples

nsym = 20
method_list = ['PUe', 'pusb', 'sar-em', 'lbe', 'pglin', 'threshold', 'oracle']
label_strats = ['S1', 'S2', 'S3', 'S4']

if 'results_artificial' not in os.listdir():
    os.mkdir('results_artificial')

for label_strat in label_strats:
    for xdistr in ['norm','unif','lognorm']:
        print('\n Distribution:', xdistr)
        for method in method_list:
            print('\n Method:', method)
            results = np.zeros((nsym,7))
            for sym in range(nsym):

                # Set the seeds
                np.random.seed(sym)
                random.seed(sym)
                km.seed(sym)

                pusb.seed(sym)
                lbe.seed(sym)
                pgl.seed(sym)
                sarem.seed(sym)
                threshold.seed(sym)
                basic.seed(sym)
                pue.seed(sym)
                
                print('|', sep=' ', end='', flush=True) 
                y, ytest, X,Xtest, prob_true, prob_true_test = generate_artificial_data_gen(n=n,p=p,b=1,xdistr=xdistr)
                n = X.shape[0]

                # Make PU data set
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
                
                if method in ['pusb', 'PUe']: #Transform data to case-control scenario
                    X = np.concatenate((X, X[s==1]))
                    s = np.concatenate((np.zeros(len(s)), np.ones(int(np.sum(s)))))

                if method == 'oracle':
                    model = PUbasic()
                    model.fit(X,y)
                else:
                    if method == 'threshold':
                        model = PUthreshold() 
                    elif method == 'sar-em':
                        model = SAREM()
                    elif method == 'PUe':
                        model = PUe()
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
                
                prob_y_test = model.predict_proba(Xtest)[:,1]
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
                results[sym,2] = tnr
                results[sym,3] = tpr
                results[sym,4] = roc_auc
                results[sym,5] = pr_auc
                results[sym,6] = bacc
                
                file_out = 'results_artificial/results_method_' + method + '_label_scheme_' + label_strat + '_xdistr_' + xdistr + ".txt"  
                np.savetxt(file_out, results)
            





