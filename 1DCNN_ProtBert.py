from transformers import BertModel, BertTokenizer
import re
import torch.nn as nn
import torch
import argparse
from sklearn import metrics
# from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import matthews_corrcoef
import numpy as np
from sklearn.metrics import confusion_matrix
import math

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
        print(x)
        x = x.permute(0, 2, 1)
        print(x)
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

print("============= TESTING ======================")
# load testing dataset
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
    raise ValueError("FASTA test file is not valid.")
print("len of test seqs ids", len(seq_ids))
print("len of test seqs", len(seqs))

# feature generation
print("=====Loading pre-trained protein language model=====")
tokenizer = BertTokenizer.from_pretrained("../Rostlab/prot_bert", do_lower_case=False, cache_dir=args.cache)
pretrain_model = BertModel.from_pretrained("../Rostlab/prot_bert", cache_dir=args.cache)
print("Done!")


def get_protein_features(seq):
    sequence_Example = ' '.join(seq)
    sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1, :]
    return last_hidden.detach()


# generate sequence feature
features = []
print("=====Generating protein sequence feature=====")
for s in seqs:
    features.append(get_protein_features(s).unsqueeze(0))
print("Done!")

# load CNN model
print("=====Loading classification model=====")
predictor = CNNOD()
if args.ligand == 'DNA':
    predictor.load_state_dict(torch.load("../weights/DNA.pth"))
else:
    raise ValueError(args.ligand)
print("Done!")

# prediction process
results = []
print(f"=====Predicting {args.ligand}-binding sites=====")
for f in features:
    out = predictor(f).squeeze(0).detach().numpy()[:, 1]
    score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
    results.append(score)
print("Done!")

print(f"=====Writing result files into=====")
with open("path of txt file to save prediction", 'w') as f:
    for i in range(len(seq_ids)):
        f.write(seq_ids[i] + '\n')
        f.write(seqs[i] + '\n')
        f.write(results[i] + '\n')
print(f"Congrats! All process done! Your result file is saved as")



################ EVALUATION ###############################################
print("============= EVALUATION ======================")
# predicted file
ab_test = open("path of txt file to save prediction", 'r').readlines()
predicts = []
for line in ab_test:
    if line.startswith(('0','1')):
        predicts.append(line.strip())
print("len of predicts", len(predicts))

# groundtruth file
lab_file = open("path of test txt file with seqs and labels", 'r').readlines()
label = []
for line in lab_file:
    if line.startswith(('0','1')):
        label.append(line.strip())
print("len of lables", len(label))

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