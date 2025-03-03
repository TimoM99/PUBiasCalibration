"""
This script runs the experiments for Q1/2.

By running experiment_nn, the script will run the experiments for the neural network classifiers on image datasets (Q2).

experiment_lr can be used to run the experiments for the logistic regression classifier on UCI datasets (Q1). However, the script experiment_multi, parallelizes this and might be preferred.
"""

import os
# Not all pytorch functions work on mps, so we set the fallback to cpu
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import time
import PUBiasCalibration.helper_files.km as km
import pandas as pd

from torchvision import transforms, datasets
import PUBiasCalibration.Models.PUSB as pusb
from PUBiasCalibration.Models.PUSB import PUSB, PUSBdeep
import PUBiasCalibration.Models.LBE as lbe
from PUBiasCalibration.Models.LBE import LBEdeep, LBE
import PUBiasCalibration.Models.PGlin as pgl
from PUBiasCalibration.Models.PGlin import PUGerychDeep, PUGerych
from PUBiasCalibration.helper_files.utils import CustomDataset, label_transform_MNIST, label_transform_CIFAR10, label_transform_USPS, label_transform_Fashion, label_transform_Alzheimer, make_binary_class, sigmoid
from torchvision.datasets import MNIST, CIFAR10, USPS, FashionMNIST
import PUBiasCalibration.Models.basic as basic
from PUBiasCalibration.Models.basic import PUbasicDeep, PUbasic
import PUBiasCalibration.Models.SAREM as sarem
from PUBiasCalibration.Models.SAREM import SAREMdeep, SAREM
import PUBiasCalibration.Models.threshold as threshold
from PUBiasCalibration.Models.threshold import PUthresholddeep, PUthreshold
from PUBiasCalibration.Models.PUe import PUedeep, PUe
import PUBiasCalibration.Models.PUe as pue
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from PIL import Image



# This experiment is not entirely deterministic because of the PUSB loss function not being deterministic.
def experiment_nn(args):

    #Set parameters for the experiment
    nsym = int(args.nsym)
    label_strat = str(args.strat)
    device = int(args.device)
    ds = str(args.ds)

    # Set seeds
    torch.cuda.empty_cache()
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 512
    # Load data
    if ds == 'MNIST':
        dims = 784
        clf = 'mlp'
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        target_transformer = label_transform_MNIST()
        
        # Load dataset once with target_transformer to fully label the dataset
        trainset = MNIST(root='./data', train=True, download=True, transform=transformer, target_transform=target_transformer)
        # Load dataset once without transforming targets, these will be transformed later in the script
        trainset_pu = MNIST(root='./data', train=True, download=True, transform=transformer)
        testset = MNIST(root='./data', train=False, download=True, transform=transformer, target_transform=target_transformer)
        
    elif ds == 'CIFAR10':
        dims = None
        clf = 'cnn'
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        target_transformer = label_transform_CIFAR10()

        trainset = CIFAR10(root='./data', train=True, download=True, transform=transformer, target_transform=target_transformer)
        trainset_pu = CIFAR10(root='./data', train=True, download=True, transform=transformer)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transformer, target_transform=target_transformer)

    elif ds == 'USPS':
        dims=256
        clf = 'mlp'
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.2469,), (0.2989,))])
        target_transformer = label_transform_USPS()

        trainset = USPS(root='./data', train=True, download=True, transform=transformer, target_transform=target_transformer)
        trainset_pu = USPS(root='./data', train=True, download=True, transform=transformer)
        testset = USPS(root='./data', train=False, download=True, transform=transformer, target_transform=target_transformer)

    elif ds == 'Fashion':
        dims=784
        clf = 'mlp'
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))])
        target_transformer = label_transform_Fashion()

        trainset = FashionMNIST(root='./data', train=True, download=True, transform=transformer, target_transform=target_transformer)
        trainset_pu = FashionMNIST(root='./data', train=True, download=True, transform=transformer)
        testset = FashionMNIST(root='./data', train=False, download=True, transform=transformer, target_transform=target_transformer)

    elif ds == 'Alzheimer':
        dims = 224
        batch_size = 32
        clf = 'resnet'
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),             # Resize images to 224x224 pixels
            transforms.ToTensor(),                     # Convert images to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        target_transformer = label_transform_Alzheimer()
        
        trainset = CustomDataset('data/MRI_split/train', transform=transformer, target_transform=target_transformer)
        trainset_pu = CustomDataset('data/MRI_split/train', transform=transformer)
        testset = CustomDataset('data/MRI_split/test', transform=transformer, target_transform=target_transformer)

    train_subset, val_subset = torch.utils.data.random_split(trainset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    
    trainloader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size, num_workers=0)
    valloader = torch.utils.data.DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Label the datasets according to the label strategy
    print('observing labels')
    if label_strat == 'S1':
        for index in range(len(trainset_pu.data)):
            img, label = trainset_pu.data[index], trainset_pu.targets[index]
            if torch.is_tensor(img):
                img = img.numpy()
            mode = None if ds in ['CIFAR10'] else 'RGB' if ds in ['Alzheimer'] else 'L'

            img = Image.fromarray(img, mode=mode)
            img = transformer(img)
            thr = np.random.uniform()
            prop_score = 0.1

            if target_transformer(label) == 1:
                if prop_score >= thr:
                    trainset_pu.targets[index] = 1
                else:
                    trainset_pu.targets[index] = 0
            else:
                trainset_pu.targets[index] = 0

    elif label_strat == 'S2':
        basic.seed(42)
        model = PUbasicDeep(clf=clf, dims=dims, device=device)
        model.fit(trainloader, valloader, epochs=100, lr=1e-5)
        model.eval()

        with torch.no_grad():
            for index in range(len(trainset_pu.data)):
                img, label = trainset_pu.data[index], trainset_pu.targets[index]
                
                if torch.is_tensor(img):
                    img = img.numpy()
                mode = None if ds in ['CIFAR10'] else 'RGB' if ds in ['Alzheimer'] else 'L'
                img = Image.fromarray(img, mode=mode)
                img = transformer(img)

                thr = np.random.uniform()
                prop_score = 0.1 * model.predict_proba(img.unsqueeze(0).to(model.device))

                if target_transformer(label) == 1:
                    if prop_score >= thr:
                        trainset_pu.targets[index] = 1
                    else:
                        trainset_pu.targets[index] = 0
                else:
                    trainset_pu.targets[index] = 0


    elif label_strat == 'S3':
        basic.seed(42)
        model = PUbasicDeep(clf=clf, dims=dims, device=device)
        model.fit(trainloader, valloader, epochs=100, lr=1e-5)
        model.eval()

        with torch.no_grad():
            for index in range(len(trainset_pu.data)):
                img, label = trainset_pu.data[index], trainset_pu.targets[index]
                
                if torch.is_tensor(img):
                    img = img.numpy()
                mode = None if ds in ['CIFAR10'] else 'RGB' if ds in ['Alzheimer'] else 'L'
                img = Image.fromarray(img, mode=mode)
                img = transformer(img)

                thr = np.random.uniform()
                prop_score = torch.sigmoid(-0.5 * model.predict_proba(img.unsqueeze(0).to(model.device)) - 1.5)

                if target_transformer(label) == 1:
                    if prop_score >= thr:
                        trainset_pu.targets[index] = 1
                    else:
                        trainset_pu.targets[index] = 0
                else:
                    trainset_pu.targets[index] = 0


    elif label_strat == 'S4':
        basic.seed(42)
        model = PUbasicDeep(clf=clf, dims=dims, device=device)
        model.fit(trainloader, valloader, epochs=100, lr=1e-5)
        model.eval()
        
        with torch.no_grad():
            for index in range(len(trainset_pu.data)):
                img, label = trainset_pu.data[index], trainset_pu.targets[index]
                
                if torch.is_tensor(img):
                    img = img.numpy()
                mode = None if ds in ['CIFAR10'] else 'RGB' if ds in ['Alzheimer'] else 'L'
                img = Image.fromarray(img, mode=mode)
                img = transformer(img)
                
                thr = np.random.uniform()
                prop_score = 0.5 * torch.sigmoid(-0.5 * torch.logit(model.predict_proba(img.unsqueeze(0).to(model.device))))

                if target_transformer(label) == 1:
                    if prop_score.item() >= thr:
                        trainset_pu.targets[index] = 1
                    else:
                        trainset_pu.targets[index] = 0
                else:
                    trainset_pu.targets[index] = 0
    
    # Making the case control dataset

    def custom_transform(img):
        if torch.is_tensor(img):
            img = img.numpy()
        mode = None if ds in ['CIFAR10'] else 'RGB' if ds in ['Alzheimer'] else 'L'

        img = Image.fromarray(img, mode=mode)
        img = transformer(img)
        return img
    
    # Apply the transformation to the subset with label 0
    transformed_data_1 = torch.stack([custom_transform(img) for img in np.array(trainset_pu.data)])

    # Apply the transformation to the subset with label 1
    positive_indices = np.array(trainset_pu.targets) == 1
    transformed_data_2 = torch.stack([custom_transform(img) for img in np.array(trainset_pu.data)[positive_indices]])

    # Concatenate the transformed tensors
    combined_data = torch.concat((transformed_data_1, transformed_data_2))

    # Make the case-control dataset
    trainset_pu_case_control = torch.utils.data.TensorDataset(combined_data, 
                                                             torch.concat((torch.zeros(len(trainset_pu.data)), torch.ones(len(np.array(trainset_pu.data)[positive_indices])))))
    train_subset_pu, val_subset_pu = torch.utils.data.random_split(trainset_pu, [0.8, 0.2], generator=torch.Generator().manual_seed(40))
    train_subset_pu_case_control, val_subset_pu_case_control = torch.utils.data.random_split(trainset_pu_case_control, [0.8, 0.2], generator=torch.Generator().manual_seed(40))

    trainloader_pu = torch.utils.data.DataLoader(train_subset_pu, batch_size=batch_size, shuffle=True, num_workers=0)
    trainloader_pu_case_control = torch.utils.data.DataLoader(train_subset_pu_case_control, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader_pu = torch.utils.data.DataLoader(val_subset_pu, batch_size=batch_size, shuffle=False, num_workers=0)
    valloader_pu_case_control = torch.utils.data.DataLoader(val_subset_pu_case_control, batch_size=batch_size, shuffle=False, num_workers=0)

    for method in ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle', 'PUe']:

        print('\n Method:', method)
        
        results = np.zeros((nsym,8))
        
        for sym in np.arange(0,nsym,1):
            print(sym)
            # Set seeds
            torch.cuda.empty_cache()
            np.random.seed(sym)
            torch.manual_seed(sym)
            torch.cuda.manual_seed(sym)
            torch.cuda.manual_seed_all(sym)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            km.seed(sym)
            

            pusb.seed(sym)
            pue.seed(sym)
            lbe.seed(sym)
            pgl.seed(sym)
            sarem.seed(sym)
            threshold.seed(sym)
            basic.seed(sym)

            if method in ['pusb', 'PUe']:
                observed_ind = [i for i, (_, y) in enumerate(train_subset_pu_case_control) if y == 1]
                unlabeled_ind = [i for i in range(len(train_subset_pu_case_control)) if i not in observed_ind]
                pos_train_subset_pu = torch.utils.data.Subset(train_subset_pu_case_control, observed_ind)
                neg_train_subset_pu = torch.utils.data.Subset(train_subset_pu_case_control, unlabeled_ind)

                sample_size = 1600
                observed_loader = torch.utils.data.DataLoader(dataset=pos_train_subset_pu, shuffle=True, batch_size=min(sample_size, len(pos_train_subset_pu)))
                unlabeled_loader = torch.utils.data.DataLoader(dataset=neg_train_subset_pu, shuffle=True, batch_size=min(sample_size, len(neg_train_subset_pu)))
                X_mixture = torch.flatten(next(iter(unlabeled_loader))[0], start_dim=1).float().numpy().astype(np.double)
                X_component = torch.flatten(next(iter(observed_loader))[0], start_dim=1).float().numpy().astype(np.double)

                km1 = km.km_default(X_mixture, X_component)
                naive_class_prior = len(pos_train_subset_pu)/len(train_subset_pu_case_control)
                est_pi = (1-naive_class_prior)*km1[1] + naive_class_prior
            
            # We track time, but it's not super accurate in this experiment as we don't limit the resources.
            start = time.time()
            y_test = np.zeros(len(testloader.dataset))
            y_prob = np.zeros(len(testloader.dataset))
            if method == 'oracle':
                model = PUbasicDeep(clf=clf, dims=dims, device=device)
                model.fit(trainloader, valloader, epochs=100, lr=1e-5)
            else: 
                if method == 'naive':
                    model = PUbasicDeep(clf=clf, dims=dims, device=device)
                elif method == 'lbe':
                    model = LBEdeep(clf=clf, dims=dims, device=device)
                elif method == 'sar-em':
                    model = SAREMdeep(clf=clf, dims=dims, device=device)
                elif method == 'threshold':
                    model = PUthresholddeep(clf=clf, dims=dims, device=device)
                elif method == 'PUe':
                    model = PUedeep(clf=clf, dims=dims, est_pi=est_pi, device=device)
                elif method == 'pusb':
                    model = PUSBdeep(clf=clf, dims=dims, class_prior=est_pi, device=device)
                elif method == 'pglin':
                    model = PUGerychDeep(clf=clf, dims=dims, device=device)
                if method in ['pusb', 'PUe']:
                    model.fit(trainloader=trainloader_pu_case_control, valloader=valloader_pu_case_control, epochs=100, lr=1e-5)
                else:
                    model.fit(trainloader=trainloader_pu, valloader=valloader_pu, epochs=100, lr=1e-5)
            end = time.time()

            for i, data in enumerate(testloader):
                inputs, labels = data[0].to(model.device), data[1]
                y_test[i*batch_size:i*batch_size + len(labels)] = labels
                y_prob[i*batch_size:i*batch_size + len(labels)] = model.predict_proba(inputs).squeeze().detach().cpu().numpy()
            
            if np.any(np.isnan(y_prob)):
                acc=0
                f1 =0
                prec=0
                recall = 0
                roc_auc=0.5
                pr_auc=0
                bacc=0
            else:
                acc = accuracy_score(y_test, np.where(y_prob>0.5,1,0))
                f1 = f1_score(y_test, np.where(y_prob>0.5,1,0))
                prec = precision_score(y_test, np.where(y_prob>0.5,1,0))
                recall = recall_score(y_test, np.where(y_prob>0.5,1,0))
                fpr_thr, tpr_thr, thr = roc_curve(y_test, y_prob, pos_label=1)
                roc_auc = auc(fpr_thr, tpr_thr)
                prec_thr, recall_thr, thr = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall_thr,prec_thr)
                bacc = balanced_accuracy_score(y_test, np.where(y_prob>0.5,1,0))
                
            results[sym,0] = acc
            results[sym,1] = f1
            results[sym,2] = prec
            results[sym,3] = recall
            results[sym,4] = roc_auc
            results[sym,5] = pr_auc
            results[sym,6] = bacc
            results[sym,7] = end - start
            
            file_out = 'results_image/results_method_' + method + '_label_scheme_' + label_strat + '_classifier_' + 'nn' + '_ds_' + ds + ".txt"  
            np.savetxt(file_out, results)

def experiment_lr(args):
    label_strat = str(args.strat)
    nsym = int(args.nsym)
    ds = str(args.ds)


    # Load data:
    df_name = 'data/' + ds + '.csv'
    df = pd.read_csv(df_name, sep=',')
    del df['BinClass']
    df = df.to_numpy()
    p = df.shape[1]-1
    Xall = df[:,0:p]
    yall = df[:,p]
    yall = make_binary_class(yall)
    
    
    for method in ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle', 'threshold_balanced']:
    # for method in ['oracle']:
        print('\n Method:', method)
        
        results = np.zeros((nsym, 8))
        
        for sym in np.arange(0, nsym, 1):
            # Set seeds
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
            prob_true = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
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
            
            if method in ['pusb', 'PUe']: #Transform data to case-control scenario
                X = np.concatenate((X, X[s==1]))
                s = np.concatenate((np.zeros(len(s)), np.ones(int(np.sum(s)))))
            
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-nsym', required=True)
    parser.add_argument('-clf', required=True)
    # parser.add_argument('-strat', required=True)
    parser.add_argument('-device', required=True)
    parser.add_argument('-ds', required=True)
    args = parser.parse_args()
    for strat in ['S1', 'S2', 'S3', 'S4']:
        args.strat = strat
        if args.clf == 'nn':
            if 'results_image' not in os.listdir():
                os.mkdir('results_image')
            experiment_nn(args)
        elif args.clf == 'lr':
            if 'results_UCI' not in os.listdir():
                os.mkdir('results_UCI')
            experiment_lr(args)