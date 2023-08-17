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
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import math


class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=2, num_dim=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim)) # random initialize as parameters

    def forward(self, inputs, targets):
        # resize inputs, delete labels with -1
        inputs = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
        # targets = targets.reshape(targets.size(0) * targets.size(1))
        ignore_idx = targets != -1
        inputs = inputs[ignore_idx]
        targets = targets[ignore_idx]
        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) # [batch, dim]
        centers_batch = self.centers.gather(0, targets_expand)
        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)  # [batch, batch, dim]
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1) # as above
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
        # for each anchor, find the hardest positive and negative (the furthest positive and nearest negative)
        # hard mining
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size): # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will causes loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# 1DCNN definition
class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)
        self.head = nn.Softmax(-1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x)

# load instructions
parse = argparse.ArgumentParser()
parse.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA',
                   default='DNA', choices=['DNA'])
parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parse.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')

args = parse.parse_args()

# parameter judge
if args.threshold > 1 or args.threshold < 0:
    raise ValueError("Threshold is out of range.")

# load training dataset
in_file = open("path to train dataset txt file with seqs and labels", 'r').readlines()
seq_ids = []
seqs = []
lab = []
for line in in_file:
    if line.startswith('>'):
        seq_ids.append(line.strip())
    elif line.startswith(('0','1')):
        lab.append(line.strip())
    else:
        seqs.append(line.strip())
if len(seq_ids) != len(seqs):
    raise ValueError("FASTA file is not valid.")
print("len of train seqs ids", len(seq_ids))
print("len of train seqs", len(seqs))

# load pretrain CNN model
print("=====Loading classification model=====")
predictor = CNNOD()
if args.ligand == 'DNA':
    predictor.load_state_dict(torch.load(".././weights/DNA.pth"))
else:
    raise ValueError(args.ligand)
print("Done!")

optimizer = torch.optim.Adam(predictor.parameters())
epoch = 10
t_criteria = TripletCenterLoss()
criterion = FocalLoss(gamma=0.7)

features = np.load("path to the train data embeddings file", allow_pickle=True)
m = torch.nn.Sigmoid()

# ################ TRAINING ###############################################
print("============= TRAINING ======================")
for e in range(epoch):
    for i in range(len(features)):
        loss_list = []
        f = features[i]
        f = torch.from_numpy(f)
        f = f.unsqueeze(0)
        out = predictor(f)
        tmp_lab = []
        for j in range(len(lab[i])):
            tmp_lab.append(int(lab[i][j]))
        tmp_lab = torch.tensor(tmp_lab)
        t_loss = t_criteria(out, tmp_lab)
        out = torch.squeeze(out, 0)
        f_loss = criterion(m(out), tmp_lab)
        loss = f_loss + ((0.1)*(t_loss))
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch ", e)
    print("loss", sum(loss_list)/len(loss_list))

# ################ TESTING ###############################################
print("============= TESTING ======================")
# load testing dataset
in_file = open("path of test txt file with seqs and labels", 'r').readlines()
seq_ids = []
seqs = []
lab = []
for line in in_file:
    if line.startswith('>'):
        seq_ids.append(line.strip())
    elif line.startswith(('0','1')):
        lab.append(line.strip())
    else:
        seqs.append(line.strip())
if len(seq_ids) != len(seqs):
    raise ValueError("FASTA test file is not valid.")
print("len of test seqs ids", len(seq_ids))
print("len of test seqs", len(seqs))

# generate sequence feature
features = np.load("path to the test data embeddings file", allow_pickle=True)
print("=====Generating protein sequence feature=====")

# prediction process
results = []
print(f"=====Predicting {args.ligand}-binding sites=====")
for f in features:
    f = torch.from_numpy(f)
    f = f.unsqueeze(0)
    out = predictor(f).squeeze(0).detach().numpy()[:, 1]
    score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
    results.append(score)
print("Done!")

# writing the results into the output file
print(f"=====Writing result files into=====")
with open("path of txt file to save prediction", 'w') as f:
    for i in range(len(seq_ids)):
        f.write(seq_ids[i] + '\n')
        f.write(seqs[i] + '\n')
        f.write(results[i] + '\n')
print(f"Congrats! All process done! Your result file is saved")


################ EVALUATION ###############################################
print("============= EVALUATION ======================")
# predicted file
ab_test = open("path of txt file to save prediction", 'r').readlines()
predicts = []
for line in ab_test:
    if line.startswith(('0','1')):
        predicts.append(line.strip())
print("len predicts", len(predicts))

# groundtruth file
lab_file = open("path of test txt file with seqs and labels", 'r').readlines()
label = []
for line in lab_file:
    if line.startswith(('0','1')):
        label.append(line.strip())
print("len label", len(label))

prec = []
recall = []
f1_weighted = []
f1_macro = []
f1_micro = []
accry = []
roc_auc = []
mcc = []
spe = []
f1 = []
dprec = []
drecall = []
df1 = []
dmcc = []
for i in range(len(label)):
    tmp_lab = []
    tmp_pred = []
    for j in range(len(label[i])):
        tmp_lab.append(int(label[i][j]))
        tmp_pred.append(int(predicts[i][j]))

    dprec.append(metrics.precision_score(np.array(tmp_lab), np.array(tmp_pred)))
    drecall.append(metrics.recall_score(np.array(tmp_lab), np.array(tmp_pred)))
    df1.append(metrics.f1_score(np.array(tmp_lab), np.array(tmp_pred)))
    dmcc.append(matthews_corrcoef(np.array(tmp_lab), np.array(tmp_pred)))

    f1_weighted.append(metrics.f1_score(np.array(tmp_lab), np.array(tmp_pred), average='weighted'))
    f1_macro.append(metrics.f1_score(np.array(tmp_lab), np.array(tmp_pred), average='macro'))
    accry.append(metrics.accuracy_score(np.array(tmp_lab), np.array(tmp_pred)))
    roc_auc.append(metrics.roc_auc_score(np.array(tmp_lab), np.array(tmp_pred)))

    tn, fp, fn, tp = confusion_matrix(np.array(tmp_lab), np.array(tmp_pred)).ravel()
    spe.append(tn / (tn + fp))
    p_pre = tp / (tp + fp)
    prec.append(p_pre)
    p_recall = tp / (tp + fn)
    recall.append(p_recall)
    p_f1 = 2 * ((p_recall * p_pre) / (p_pre + p_recall))
    f1_micro.append(p_f1)
    p_mcc = (tp * tn - fn * fp) / math.sqrt((tp + fp) * (fp + fn) * (tn + fp) * (tn + fn))
    mcc.append(p_mcc)
    f1.append(metrics.f1_score(np.array(tmp_lab), np.array(tmp_pred)))

print("Test Precision ", sum(prec)/len(prec))
print("Test Recall ", sum(recall)/len(recall))
print("Test F1-Weighted ", sum(f1_weighted)/len(f1_weighted))
print("Test F1-Macro ", sum(f1_macro)/len(f1_macro))
print("Test F1-Binary ", sum(f1_micro)/len(f1_micro))
print("Test sklearn F1-Binary ", sum(f1)/len(f1))
print("Test SKlearn accuracy", sum(accry)/len(accry))
print("Test roc auc", sum(roc_auc)/len(roc_auc))
print("Test mcc", sum(mcc)/len(mcc))
print("Test Specificity", sum(spe)/len(spe))
print("precison sklearn", sum(dprec)/len(dprec))
print("recall sklearn", sum(drecall)/len(drecall))
print("f1 sklearn", sum(df1)/len(df1))
print("mcc sklearn", sum(dmcc)/len(dmcc))