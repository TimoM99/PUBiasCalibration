from calendar import c
from locale import normalize
from math import isnan, log, nan
from xml.etree.ElementTree import C14NWriterTarget
from scipy import optimize
import torch
import tqdm
import numpy as np
import src.helper_files.km as km

from torch import nn
from src.helper_files.classifiers import LR, MLPReLU, FullCNN, Resnet
from src.threshold_optimizer import ThresholdOptimizer
from src.helper_files.utils import EarlyStopping, sigmoid
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

def seed(seed):
    torch.manual_seed(seed)
    km.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class CustomLoss_e(nn.Module):
    def __init__(self, n_p, n_U, alpha) -> None:
        super().__init__()
        self.n_p = n_p
        self.n_U = n_U
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        loss1 = torch.sum(-1/(self.n_p +self.n_U)*y_true*torch.log(torch.sigmoid(y_pred)))
        loss2 = torch.sum(-1/(self.n_p + self.n_U)*(1-y_true)*torch.log(1-torch.sigmoid(y_pred)))
        regularisation = self.alpha*torch.abs(torch.sum(torch.sigmoid(y_pred)) - self.n_p)
        loss = loss1 + loss2 + regularisation
        # if torch.isnan(loss):
        #     print(loss1,loss2,regularisation)
        return loss
    
class CustomLoss_clf(nn.Module):
    def __init__(self, pi) -> None:
        super().__init__()
        self.pi = pi

    def forward(self, y_pred, y_true, prop_scores):
        loss1 = torch.sum(-1/prop_scores*y_true*torch.log(torch.sigmoid(y_pred)))
        loss2 = torch.sum(-(1-y_true)*torch.log(1-torch.sigmoid(y_pred)))
        loss3 = torch.sum(-1/prop_scores*y_true*torch.log(1-torch.sigmoid(y_pred)))
        
        return self.pi*loss1 + loss2 - self.pi*loss3

# Model definition
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)


    def forward(self, x):
        return self.linear(x)
    

class PUe(BaseEstimator):
   
    def __init__(self):
        self.e = None
        self.clf = None
        
    def fit(self, X, s):
        X_mixture = X[np.where(s==0)[0],:]
        X_component = X[np.where(s==1)[0],:]
        km1 = km.km_default(X_mixture, X_component)
        est_pi = (1-np.mean(s))*km1[1] +  np.mean(s)

        X = torch.from_numpy(X).float()
        s = torch.from_numpy(s).float()
        self.e = LogisticRegressionModel(X.shape[1])
        criterion = CustomLoss_e(n_p=torch.sum(s), n_U=s.shape[0] - torch.sum(s), alpha=15)
        optimizer = torch.optim.Adam(self.e.parameters(), lr=1e-3)

        for epoch in range(20):
            optimizer.zero_grad()
            outputs = self.e(X)
            loss = criterion(outputs, s)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            prop_scores = torch.sigmoid(self.e(X))
        normalized_prop_scores = torch.sum(torch.where(s == 1, 1/prop_scores, 0))*prop_scores

        self.clf = LogisticRegressionModel(X.shape[1])
        criterion = CustomLoss_clf(pi=est_pi)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=1e-3)

        for epoch in range(20):
            optimizer.zero_grad()
            outputs = self.clf(X)
            loss = criterion(outputs, s, normalized_prop_scores)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
       return np.where(self.predict_proba(X)>0.5,1,0) 
        
    def predict_proba(self, Xtest):
        with torch.no_grad():
            Xtest = torch.from_numpy(Xtest).float()
            scores = self.clf(Xtest)
            probs = torch.sigmoid(scores).squeeze().numpy()
            return np.array(list(zip(1 - probs, probs)))
        

class PUedeep(nn.Module):
    def __init__(self, clf, dims=None, est_pi=None, device=0) -> None:
        super().__init__()
        self.device = "mps" if getattr(torch, 'has_mps', False) else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == 'lr' or clf == 'mlp':
            assert dims != None, 'Classifier type {} requires specifying the dimensionality of the data.'.format(clf)

        if clf == 'lr':
            self.e = LR(dims=dims).to(self.device)
            self.clf = LR(dims=dims).to(self.device)
        elif clf == 'mlp':
            self.e = MLPReLU(dims=dims).to(self.device)
            self.clf = MLPReLU(dims=dims).to(self.device)
        elif clf == 'cnn':
            self.e = FullCNN().to(self.device)
            self.clf = FullCNN().to(self.device)
        elif clf == 'resnet':
            self.e = Resnet().to(self.device)
            self.clf = Resnet().to(self.device)

        self.pi = est_pi

        
    def predict_proba(self, x):
        # assert self.threshold != None, 'The model has to be fit before predictions can be made.'
        with torch.no_grad():
            scores = self.clf(x, probabilistic=False)
            return torch.sigmoid(scores)
        
    def calculate_prop_scores(self, trainloader):
        with torch.no_grad():
            prop_scores = []
            s = []
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                pred_y = torch.sigmoid(self.e(inputs, probabilistic=False))
                prop_scores.append(pred_y)
                s.append(labels)
            prop_scores = torch.cat(prop_scores)
            s = torch.cat(s)
        
        self.normalizing_factor = torch.sum(torch.where(s == 1, 1/prop_scores, 0))
        


    def fit(self, trainloader, valloader, epochs, lr=1e-3):
        optimizer_e = torch.optim.Adam(self.e.parameters(), lr=lr)
        optimizer_clf = torch.optim.Adam(self.clf.parameters(), lr=lr)
        
        count = 0
        count_positive = 0
        for data, labels in trainloader:
            count_positive += torch.sum(labels)
            count += len(labels)

        criterion_e = CustomLoss_e(n_p=count_positive, n_U=count-count_positive, alpha=15)
        criterion_clf = CustomLoss_clf(self.pi)
        
        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:
                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer_e.zero_grad()
                # print(inputs)
                # print(inputs.dtype)
                outputs = self.e(inputs, probabilistic=False)
                loss = criterion_e(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer_e.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    v_loss = 0

                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                        pred_y = self.e(inputs, probabilistic=False)
                        v_loss += criterion_e(pred_y, labels.unsqueeze(1).float()).item()

                    v_loss = v_loss/(j + 1)

                    if es(self.e, v_loss):
                        done = True

                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss:}")
            
            if done == True:
                break

        self.calculate_prop_scores(trainloader)
        
        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    prop_scores = torch.sigmoid(self.e(inputs, probabilistic=False))
                    normalized_prop_scores = self.normalizing_factor*prop_scores
                optimizer_clf.zero_grad()
                outputs = self.clf(inputs, probabilistic=False)
                loss = criterion_clf(outputs, labels.unsqueeze(1).float(), normalized_prop_scores)
                loss.backward()
                optimizer_clf.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    v_loss = 0

                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                        with torch.no_grad():
                            prop_scores = torch.sigmoid(self.e(inputs, probabilistic=False))
                            normalized_prop_scores = self.normalizing_factor*prop_scores
                        pred_y = self.clf(inputs, probabilistic=False)
                        v_loss += criterion_clf(pred_y, labels.unsqueeze(1).float(), normalized_prop_scores).item()

                    v_loss = v_loss/(j + 1)

                    if es(self.clf, v_loss):
                        done = True

                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss:}")
            
            if done == True:
                break
        
        