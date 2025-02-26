# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:31:36 2023

@author: teiss
"""
import numpy as np
from src.helper_files.utils import sigmoid
from sklearn.preprocessing import scale

def generate_artificial_data_discr(prior=0.5,n=2000,p=10,xdistr="norm"): 
    y = np.zeros(n)
    ytest = np.zeros(n)
    for i in np.arange(0,n,1):
        y[i] = np.random.binomial(1, prior, size=1)
        ytest[i] = np.random.binomial(1, prior, size=1)

    
    X=np.zeros((n,p))
    ind1 = np.where(y==1)[0]
    ind0 = np.where(y==0)[0]
    
    Xtest=np.zeros((n,p))
    ind1test = np.where(ytest==1)[0]
    ind0test = np.where(ytest==0)[0]
    
    
    if xdistr=='norm':
        for j in np.arange(0,p,1):
                X[ind0,j]=np.random.normal(loc=0,scale=1,size=len(ind0))
                X[ind1,j]=np.random.normal(loc=1,scale=1,size=len(ind1))
                Xtest[ind0,j]=np.random.normal(loc=0,scale=1,size=len(ind0test))
                Xtest[ind1,j]=np.random.normal(loc=1,scale=1,size=len(ind1test))
    elif xdistr=='unif':
        for j in np.arange(0,p,1):
                X[ind0,j]=np.random.uniform(-1,1,size=len(ind0))
                X[ind1,j]=np.random.uniform(0,2,size=len(ind1))
                Xtest[ind0,j]=np.random.uniform(-1,1,size=len(ind0test))
                Xtest[ind1,j]=np.random.uniform(0,2,size=len(ind0test))   
    elif xdistr=='lognorm':
        for j in np.arange(0,p,1):
                X[ind0,j]=np.exp(np.random.normal(loc=0,scale=1,size=len(ind0)))
                X[ind1,j]=np.exp(np.random.normal(loc=1,scale=1,size=len(ind1)))
                Xtest[ind0,j]=np.exp(np.random.normal(loc=0,scale=1,size=len(ind0test)))
                Xtest[ind1,j]=np.exp(np.random.normal(loc=1,scale=1,size=len(ind1test)))      
    else:
         print('Argument xdistr is not defined')       
        
    return y,ytest,X,Xtest   


def generate_artificial_data_gen(n=2000,p=10,b=1,xdistr="norm"):


    X=np.zeros((n,p))
    Xtest=np.zeros((n,p))

    if xdistr=='norm':
        for j in np.arange(0,p,1):
                X[:,j]=np.random.normal(loc=0,scale=1,size=n)
                Xtest[:,j]=np.random.normal(loc=0,scale=1,size=n)
    elif xdistr=='unif':
        for j in np.arange(0,p,1):
                X[:,j]=np.random.uniform(-1,1,size=n)
                Xtest[:,j]=np.random.uniform(-1,1,size=n)   
    elif xdistr=='lognorm':
        for j in np.arange(0,p,1):
                X[:,j]=np.exp(np.random.normal(loc=0,scale=1,size=n))
                Xtest[:,j]=np.exp(np.random.normal(loc=0,scale=1,size=n))      
    else:
         print('Argument xdistr is not defined')       
        
    X = scale(X, axis=0, with_mean=True, with_std=True)    
    Xtest = scale(Xtest, axis=0, with_mean=True, with_std=True)    
    
    
    beta_true = np.zeros(p)
    beta_true[0:5]= b
    eta = np.dot(X,beta_true)
    prob_true = sigmoid(eta)
    eta_test = np.dot(Xtest,beta_true)
    prob_true_test = sigmoid(eta_test)    
    y = np.zeros(n)
    ytest = np.zeros(n)
    for i in np.arange(0,n,1):
        y[i] = np.random.binomial(1, prob_true[i], size=1)
        ytest[i] = np.random.binomial(1, prob_true_test[i], size=1)    
        
        
    return y,ytest,X,Xtest,prob_true,prob_true_test    