from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd


##ML imports
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rdkit.Chem import Draw
from scipy.spatial import distance

import sys

#turn off scikit warning to clean output
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def generateMolClassifiedImages(mols, targets, ids, size=(244,244), folder='molImages'):
	positiveDir = os.path.join(folder,'1')
	negativeDir = os.path.join(folder,'0')
	if not os.path.exists(positiveDir):
		os.makedirs(positiveDir)
	if not os.path.exists(negativeDir):
		os.makedirs(negativeDir)
	for mol, target, molID in zip(mols, targets, ids):
		if target == 1 :
			Draw.MolToFile(mol, os.path.join(positiveDir,'{0}.png'.format(molID)), size=size)
		elif target == 0 :
			Draw.MolToFile(mol, os.path.join(negativeDir,'{0}.png'.format(molID)), size=size)

def computeWeightsForTarget(class_splits, target):
    trainCountTotal = class_splits[target]['trainCount'][2]
    validCountTotal = class_splits[target]['validCount'][2]
    return {
        'train': [class_splits[target]['trainCount'][1]/trainCountTotal,
                class_splits[target]['trainCount'][0]/trainCountTotal],
        'val': [class_splits[target]['validCount'][1]/validCountTotal,
                class_splits[target]['validCount'][0]/validCountTotal]
    }

def generateDataWithTargetAndFingerprint(class_splits, target, fingerprint, length=512):
    return

def plot_TSNE(tsne_embedding, dataSet):
    f, axs = plt.subplots(5,3,figsize=(10,10))
    axs[0,1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c='blue', s=0.02)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,1].set_title('All Molecules')
    axs[0,0].axis('off')
    axs[0,2].axis('off')
    for i in np.arange(1,13):
        y = dataSet['target{0}'.format(i)]
        ax = axs[int((i-1)/3 + 1), (i-1) % 3]
        argPos = np.argwhere(y==1)
        #ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c='blue', s=0.04)
        ax.scatter(tsne_embedding[argPos, 0], tsne_embedding[argPos, 1], c='red', s=0.24)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Target {0}'.format(i))
    plt.tight_layout()
    plt.show()


def computeCustomSimilarityMatrix(dataSet):
    russelRaoMatrix = np.ones((12,12))
    for i in np.arange(12):
       for j in np.arange(i+1, 12):
            firstTarget = 'target{0}'.format(i+1)
            secondTarget = 'target{0}'.format(j+1)
            targetColumns = dataSet[[firstTarget, secondTarget]].dropna()
            u = np.array(list(targetColumns[firstTarget]))
            v = np.array(list(targetColumns[secondTarget]))
            sim  = 2 * np.sum((u + v) == 2) / (np.sum(u)+np.sum(v))
            russelRaoMatrix[i,j] =  sim
            russelRaoMatrix[j,i] =  sim
    return russelRaoMatrix

def train_LR(X, y, X_valid, y_valid):
    result = {}
    lr = LogisticRegression()
    lr_parameters = {
                  'C':[0.001,0.01,0.1,1],
                  'class_weight': ['balanced'],
                  'solver': ['lbfgs'],
                  'max_iter': [500]}
    clf = GridSearchCV(lr, lr_parameters, cv=3, scoring='f1', n_jobs=-1, return_train_score=True).fit(X,y)
    result['train f1'] = np.mean(clf.cv_results_['mean_train_score'])
    result['test f1'] = np.mean(clf.cv_results_['mean_test_score'])
    best = clf.best_estimator_
    best.fit(X, y)
    y_pred = best.predict(X_valid)
    result['valid f1'] = f1_score(y_valid, y_pred)
    result['confusion matrix'] = confusion_matrix(y_valid, y_pred)
    result['best'] = best
    return result

def train_RTF(X, y, X_valid, y_valid):
    result = {}
    rtf = RandomForestClassifier()
    rtf_parameters = {
                  'n_estimators':[100,400],
                  'max_depth':  [10,80],
                  'class_weight': ['balanced']
                  }
    clf = GridSearchCV(rtf, rtf_parameters, cv=3, scoring='f1', n_jobs=-1, return_train_score=True).fit(X,y)
    result['train f1'] = np.mean(clf.cv_results_['mean_train_score'])
    result['test f1'] = np.mean(clf.cv_results_['mean_test_score'])
    best = clf.best_estimator_
    best.fit(X, y)
    y_pred = best.predict(X_valid)
    result['valid f1'] = f1_score(y_valid, y_pred)
    result['confusion matrix'] = confusion_matrix(y_valid, y_pred)
    result['best'] = best
    return result

def train_ADA(X, y, X_valid, y_valid):
    result = {}
    ada = AdaBoostClassifier()
    ada_parameters = {
              'n_estimators':[100,400],
              'learning_rate': [0.001,0.1]}
    clf = GridSearchCV(ada, ada_parameters, cv=3, scoring='f1', n_jobs=-1, return_train_score=True).fit(X,y)
    result['train f1'] = np.mean(clf.cv_results_['mean_train_score'])
    result['test f1'] = np.mean(clf.cv_results_['mean_test_score'])
    best = clf.best_estimator_
    best.fit(X, y)
    y_pred = best.predict(X_valid)
    result['valid f1'] = f1_score(y_valid, y_pred)
    result['confusion matrix'] = confusion_matrix(y_valid, y_pred)
    result['best'] = best
    return result

def format_results_table(results):
    columnHeader = pd.MultiIndex.from_product([['target{0}'.format(x) for x in range(1,13)],
                                     ['train','test','valid']],
                                    names=['Target','Dataset'])
    rowHeader = pd.MultiIndex.from_product([['LR','RTF','ADA'],
                                     ['256', '512', '1024', '2048','4096']],
                                    names=['Model','FP Length'])
    lengths = [256, 512, 1024, 2048,4096]
    results_table = np.zeros((3*5,12*3))
    for t in range(1,13):
        for l in range(5):
            results_table[l,(t-1)*3]= results['lr'][t][l]['train f1']
            results_table[l,(t-1)*3+1]= results['lr'][t][l]['test f1']
            results_table[l,(t-1)*3+2]= results['lr'][t][l]['valid f1']

            results_table[l+5,(t-1)*3]= results['rtf'][t][l]['train f1']
            results_table[l+5,(t-1)*3+1]= results['rtf'][t][l]['test f1']
            results_table[l+5,(t-1)*3+2]= results['rtf'][t][l]['valid f1']

            results_table[l+10,(t-1)*3]= results['ada'][t][l]['train f1']
            results_table[l+10,(t-1)*3+1]= results['ada'][t][l]['test f1']
            results_table[l+10,(t-1)*3+2]= results['ada'][t][l]['valid f1']

    return pd.DataFrame(results_table, index=rowHeader, columns=columnHeader)

def print_CMs(results, target, lengthIndex):
    print('LR:')
    print(pd.DataFrame(results['lr'][target][lengthIndex]['confusion matrix'], 
                   index=['true:0', 'true:1'], 
                   columns=['pred:0', 'pred:1']))
    print()
    print('RTF:')
    print(pd.DataFrame(results['rtf'][target][lengthIndex]['confusion matrix'], 
                   index=['true:0', 'true:1'], 
                   columns=['pred:0', 'pred:1']))
    print()
    print('ADA:')
    print(pd.DataFrame(results['ada'][target][lengthIndex]['confusion matrix'], 
                   index=['true:0', 'true:1'], 
                   columns=['pred:0', 'pred:1']))
    


######
# The following code is highly inspired from the pytorch tutorial.
# I am currently learning how to use pytorch and wanted to make use of it in this
# exercise
# Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# License: BSD
# Author: Sasank Chilamkurthy
######
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(dataloaders, model, criterion, optimizer, scheduler, dataset_sizes, num_epochs=10, device="cpu"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    results = {
        'train': [],
        'val': []
    }


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            allLabels = []
            allPreds = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                allLabels.extend(labels.data)
                allPreds.extend(preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(allLabels, allPreds)

            print('{} Loss: {:.4f} Acc: {:.4f} F1:{:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))
            results[phase].append([epoch_loss, epoch_acc, epoch_f1])
            # deep copy the model based on f1 score
            if phase == 'val' and epoch_acc > best_acc:
                best_f1 = epoch_f1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results


def visualize_model(model, dataloaders, class_names, device="cpu", num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)