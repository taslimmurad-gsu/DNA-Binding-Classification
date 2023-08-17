import numpy as np
from transformers import BertModel, BertTokenizer
import re
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from sklearn import metrics
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

##########################################################
############## Start of ML Classifiers ###################
##########################################################
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    NB_acc = metrics.accuracy_score(y_test, y_pred)
    NB_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    NB_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    NB_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    NB_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    NB_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [NB_acc, NB_prec, NB_recall, NB_f1_weighted, NB_f1_macro, NB_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)

##########################  SVM Classifier  ################################
def svm_fun(X_train, y_train, X_test, y_test):
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    svm_acc = metrics.accuracy_score(y_test, y_pred)
    svm_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    svm_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    svm_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    svm_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    svm_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [svm_acc, svm_prec, svm_recall, svm_f1_weighted, svm_f1_macro, svm_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)

##########################  MLP Classifier  ################################
def mlp_fun(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test_2 = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test_2)
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
    MLP_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    MLP_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    MLP_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    MLP_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [MLP_acc, MLP_prec, MLP_recall, MLP_f1_weighted, MLP_f1_macro, MLP_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)

##########################  knn Classifier  ################################
def knn_fun(X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    knn_acc = metrics.accuracy_score(y_test, y_pred)
    knn_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    knn_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    knn_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    knn_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    knn_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [knn_acc, knn_prec, knn_recall, knn_f1_weighted, knn_f1_macro, knn_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)

##########################  Random Forest Classifier  ################################
def rf_fun(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    fr_acc = metrics.accuracy_score(y_test, y_pred)
    fr_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    fr_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    fr_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    fr_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    fr_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [fr_acc, fr_prec, fr_recall, fr_f1_weighted, fr_f1_macro, fr_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)


##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train, y_train, X_test, y_test):
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    LR_acc = metrics.accuracy_score(y_test, y_pred)
    LR_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    LR_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    LR_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    LR_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    LR_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [LR_acc, LR_prec, LR_recall, LR_f1_weighted, LR_f1_macro, LR_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)


def fun_decision_tree(X_train, y_train, X_test, y_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dt_acc = metrics.accuracy_score(y_test, y_pred)
    dt_prec = metrics.precision_score(y_test, y_pred, average='weighted')
    dt_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    dt_f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    dt_f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    dt_f1_micro = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    check = [dt_acc, dt_prec, dt_recall, dt_f1_weighted, dt_f1_macro, dt_f1_micro, roc_auc, mcc, prec, recall, specificity]
    return (check)

X_train = np.load("path to xtrain.npy", allow_pickle=True)
X_test = np.load("path to xtest.npy", allow_pickle=True)
y_train = np.load("path to ytrain.npy", allow_pickle=True)
y_test = np.load("path to ytest.npy", allow_pickle=True)

print(X_test)

print("accuray, weight-precision, weight-recall, f1-weighted, f1-macro, f1-binary, roc-auc, mcc, prec, rec, specificity")

gauu_nb_return = gaus_nb_fun(X_train, y_train, X_test, y_test)
print("NB")
print(gauu_nb_return)

mlp_return = mlp_fun(X_train, y_train, X_test, y_test)
print("MLP")
print(mlp_return)

knn_return = knn_fun(X_train, y_train, X_test, y_test)
print("KNN")
print(knn_return)

rf_return = rf_fun(X_train, y_train, X_test, y_test)
print("RF")
print(rf_return)

lr_return = lr_fun(X_train, y_train, X_test, y_test)
print("LR")
print(lr_return)

dt_return = fun_decision_tree(X_train, y_train, X_test, y_test)
print("DT")
print(dt_return)
