# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:57:46 2023

@author: teiss
"""
import os
import random
import km
# os.chdir("C:\\Users\\teiss\\Dropbox\\pu_learn")


from Models.PGlin import PUGerych
from Models.SAREM import SAREM
from Models.LBE import LBE
from Models.basic import PUbasic
from Models.threshold import PUthreshold
from Models.PUSB import PUSB
from artificial import generate_artificial_data_gen


import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from scikeras.wrappers import KerasClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc, roc_curve, precision_recall_curve, precision_score, recall_score


# from sarpu.pu_learning import pu_learn_sar_em
# from lbe.LBE_changed import lbe_train, lbe_predict_proba
from utils import sigmoid

#Set parameters:
p=10
n=2000    
par = 0.1

label_scheme= 'scar'
#label_scheme= 'prop1'
#label_scheme= 'prop2'
nsym = 20
classifier = "logistic"
method_list = ['threshold']

for label_strat in ['S4']:
    for xdistr in ['norm','unif','lognorm']:
        print('\n Distribution:', xdistr)
        for method in method_list:
            print('\n Method:', method)
            results = np.zeros((nsym,7))
            for sym in range(nsym):
                np.random.seed(sym)
                random.seed(sym)
                print('|', sep=' ', end='', flush=True) 
                y, ytest, X,Xtest, prob_true, prob_true_test = generate_artificial_data_gen(n=n,p=p,b=1,xdistr=xdistr)
                # X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=0.25, random_state=sym)
                n = X.shape[0]

                # prob_true = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
                # prob_true[np.where(prob_true==1)] = 0.999
                # prob_true[np.where(prob_true==0)] = 0.001
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
        
            
            
            # for sym in np.arange(0,nsym,1):
            
            #     print('|', sep=' ', end='', flush=True) 
                
            #     # Generate artificial data:
            #     y,ytest,X,Xtest,prob_true,prob_true_test = generate_artificial_data_gen(n=n,p=p,b=1,xdistr=xdistr)
                
            #     #Create PU dataset:
            #     s, ex_true = make_pu_labels(X,y,c=par,prob_true=prob_true,label_scheme=label_scheme,k=par)
                
            #     # if classifier=='nn':
            #     #     def create_network(p):
            #     #         model = Sequential()
            #     #         model.add(Dense(12, input_shape=(p,), activation='relu'))
            #     #         model.add(Dense(8, activation='relu'))
            #     #         model.add(Dense(1, activation='sigmoid'))
            #     #         model.compile(loss='binary_crossentropy', optimizer='adam')
            #     #         return model
            #     #     clf = KerasClassifier(create_network(p), epochs=100, verbose=0)
            #     if classifier=='logistic':    
            #         clf = LogisticRegression();
            #     else:
            #         raise Exception('classifier not found!')
                
                
                
            #     if method=='naive':
            #         model = PUbasic(clf)
            #         model.fit(X,s)
            #         prob_y_test = model.predict_proba(Xtest)[:,1]
            #     elif method=='oracle':
            #         model= PUbasic(clf)
            #         model.fit(X,y)
            #         prob_y_test = model.predict_proba(Xtest)[:,1]
            #     elif method=='weighted':
            #         model = PUweighted(clf,ex=ex_true) 
            #         model.fit(X,s)
            #         prob_y_test = model.predict_proba(Xtest)[:,1]
            #     elif method=='em':
            #         model, e_model, info = pu_learn_sar_em(X, s, range(X.shape[1]))
            #         prob_y_test = model.predict_proba(Xtest) 
            #     elif method=='threshold':
            #         model = PUthreshold(clf,score_type='mi') 
            #         model.fit(X,s)
            #         prob_y_test = model.predict_proba(Xtest)[:,1]            
            #     elif method=='threshold_weighted':
            #         model = PUthresholdWeighted(clf) 
            #         model.fit(X,s)
            #         prob_y_test = model.predict_proba(Xtest)[:,1]
            #     elif method=='lbe':
            #         model=lbe_train(X,s,kind='LR',epochs=250)
            #         prob_y_test = lbe_predict_proba(model,Xtest)
            #     else:
            #         raise Exception('method not found!')
                
                # if np.any(np.isnan(prob_y_test)):
                #     acc=0
                #     f1 =0
                #     prec=0
                #     recall = 0
                #     roc_auc=0.5
                #     prec_auc=0
                # else:
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
                
                file_out = 'Library/results_artificial/results_method_' + method + '_label_scheme_'+label_strat + '_xdistr_' + xdistr + ".txt"  
                np.savetxt(file_out, results)
            





