
import torch
import tqdm

from sklearn.base import BaseEstimator
from torch import nn
from src.helper_files.classifiers import MLPReLU, FullCNN, LR
from src.helper_files.utils import EarlyStopping
from sklearn.linear_model import LogisticRegression

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PUbasic(BaseEstimator):
   
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
    def fit(self, X, y):
        self.clf.fit(X,y)
        return self

    def predict(self, X):
        return self.clf.predict()
        
    def predict_proba(self, Xtest):
        return self.clf.predict_proba(Xtest)   

class PUbasicDeep(nn.Module):
   
    def __init__(self, clf, dims=None, device=0) -> None:
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

    def predict_proba(self, x):
        return self.clf(x, probabilistic=True)
        

    def fit(self, trainloader, valloader, epochs, lr=1e-3):
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        
        criterion = nn.BCEWithLogitsLoss()
        
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
                    v_loss = 0
                    
                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                        pred_y = self.clf(inputs, probabilistic=False)
                        v_loss += criterion(pred_y, labels.unsqueeze(1).float()).item()
                    v_loss = v_loss/(j + 1)
                    
                    if es(self.clf, v_loss):
                        done = True
                    
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
            
            if done == True:
                break