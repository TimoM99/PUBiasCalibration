import torch
import tqdm
import numpy as np

from torch import nn
from src.helper_files.utils import EarlyStopping
from src.helper_files.classifiers import LR, MLPReLU, FullCNN
from sklearn.base import BaseEstimator 
from sklearn.linear_model import LogisticRegression

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class PUGerych(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.max_sx = 1

    def fit(self, X, s):
        self.clf.fit(X,s)
        sx = self.clf.predict_proba(X)[:,1]
        self.max_sx = np.max(sx)

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1]>0.5,1,0) 
        
    def predict_proba(self, Xtest):
        sx_test = self.clf.predict_proba(Xtest)[:, 1]
        ex_test = np.sqrt(self.max_sx*sx_test)
        yx_test = (1/ex_test) * sx_test
        yx_test[np.where(yx_test>1)]=1
        
        return np.array(list(zip(1 - yx_test, yx_test))) 
    

class PUGerychDeep(nn.Module):
    def __init__(self, clf, dims, device=0):
        super().__init__()
        self.device = "mps" if getattr(torch, 'has_mps', False) else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == 'lr' or clf == 'mlp':
            assert dims != None, 'Classifier type {} requires specifying the dimensionality of the data.'.format(clf)
        
        if clf == 'lr':
            self.clf = LR(dims=dims).to(self.device)
        elif clf == 'mlp':
            self.clf = MLPReLU(dims=dims).to(self.device)
        elif clf == 'cnn':
            self.clf = FullCNN().to(self.device)
        
        self.max_sx = 0

    def predict_proba(self, x):
        with torch.no_grad():
            sx = self.clf(x, probabilistic=True)
            ex = torch.sqrt(self.max_sx * sx)
            yx = (1/ex) * sx
            return torch.clip(yx, max=1)
        
    def fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                
                outputs = self.clf(inputs, probabilistic=False)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                            pred_y = self.clf(inputs, probabilistic=False)
                            v_loss += criterion(pred_y, labels.unsqueeze(1).float()).item()
                    v_loss = v_loss/(j + 1)
                    if es(self.clf, v_loss):
                        done = True
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                    self.train()
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
            if done == True:
                break
        
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                inputs = data[0].to(self.device)
                s = self.clf(inputs, probabilistic=True)
                self.max_sx = max(self.max_sx, max(s))
        self.train()