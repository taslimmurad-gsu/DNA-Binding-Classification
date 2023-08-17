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

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

# load instructions
parse = argparse.ArgumentParser()
parse.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA',
                   default='DNA', choices=['DNA'])
parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parse.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')

args = parse.parse_args()

#########################################################
############# Start of data processing ####################
#########################################################
# load training dataset
tain_file = open("path to train dataset txt file with seqs and labels", 'r').readlines()
train_seq_ids = []
train_seqs = []
train_lab = []
for line in tain_file:
    if line.startswith('>'):
        train_seq_ids.append(line.strip())
    elif line.startswith(('0','1')):
        train_lab.append(line.strip())
    else:
        train_seqs.append(line.strip())
if len(train_seq_ids) != len(train_seqs):
    raise ValueError("Train file is not valid.")
print("len train_lab", len(train_lab))
print("len train_lab[0]", len(train_lab[0]))
print("len train_lab[0][0]", len(train_lab[0][0]))

# load testing dataset
test_file = open("path to test dataset txt file with seqs and labels", 'r').readlines()
test_seq_ids = []
test_seqs = []
test_lab = []
for line in test_file:
    if line.startswith('>'):
        test_seq_ids.append(line.strip())
    elif line.startswith(('0','1')):
        test_lab.append(line.strip())
    else:
        test_seqs.append(line.strip())
if len(test_seq_ids) != len(test_seqs):
    raise ValueError("Test file is not valid.")
print("len test_lab", len(test_lab))
print("len test_lab[0]", len(test_lab[0]))
print("len test_lab[0][0]", len(test_lab[0][0]))

# feature generation
print("=====Loading pre-trained protein language model=====")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir=args.cache)
pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=args.cache)
print("Done!")

def get_protein_features(seq):
    sequence_Example = ' '.join(seq)
    sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1, :]
    return last_hidden.detach()


# #generate sequence feature
features = []
print("=====Generating protein sequence feature=====")
for s in train_seqs:
    features.append(get_protein_features(s))
print("Train Done!")
print("len features", len(features))
print("len features[0]", len(features[0]))
print("len features[0][0]", len(features[0][0]))

test_features = []
for s in test_seqs:
    test_features.append(get_protein_features(s))
print("Test Done!")
print("len test_features", len(test_features))
print("len test_features[0]", len(test_features[0]))
print("len test_features[0][0]", len(test_features[0][0]))

X_train = []
y_train = []
X_test  = []
y_test = []

for i in range(len(features)):
    f = features[i]
    for j in range(len(train_lab[i])):
        y_train.append(int(train_lab[i][j]))
        X_train.append(f[j])

for i in range(len(test_features)):
    f = test_features[i]
    for j in range(len(test_lab[i])):
        y_test.append(int(test_lab[i][j]))
        X_test.append(f[j])

print("X train size", len(X_train))
print("y train size", len(y_train))
print("x test size", len(X_test))
print("y test size", len(y_test))

np.save("path of xtest.npy", np.array(X_test))
np.save("path of ytest.npy", np.array(y_test))
np.save("path of xtrain.npy", np.array(X_train))
np.save("path of ytrain.npy", np.array(y_train))
print("DONE")
#######################################################
########### End of data processing ####################
#######################################################