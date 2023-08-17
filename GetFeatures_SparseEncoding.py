from itertools import product
import numpy as np
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def build_kmers(sequence, ksize):
    seq = sequence
    Kmer = ksize
    L = len(seq)
    k_mers_final = []
    for i in range(0, L - Kmer + 1):
        sub_f = seq[i:i + Kmer]
        k_mers_final.append(sub_f)
    tmp = []
    for i in range(len(seq), len(seq)-8, -1):
        sub_f = seq[i - Kmer:i]
        sub_f = ''.join(reversed(sub_f))
        tmp.append(sub_f)
    for i in range(len(tmp)-1, -1, -1):
        k_mers_final.append(tmp[i])
    return k_mers_final

in_file = open("path to dataset txt file with seqs and labels", 'r').readlines()
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

enc = OneHotEncoder(handle_unknown='ignore')
data_final = []

for i in range(len(seqs)):
    data = []
    kmer = build_kmers(seqs[i], 9)
    for i in range(len(kmer)):
        tmp_data = pd.DataFrame(list(kmer[i]))
        enc.fit(tmp_data)
        aa = enc.transform(tmp_data).toarray()
        asd = list(np.concatenate(aa).flat)
        data.append(asd)
    data_final.append(data)

print("len of ohe", len(data_final))
print("len of labels", len(lab))
np.save("./kmerOHE_DNA_129_test.npy", np.array(data_final))
print("done")

######################################################################
###### uncomment this block for only sparse encoding #################
######################################################################
features = np.array(data_final)
arr_len = 1024
X_padded_final = []
for i in range(len(features)):
    X_padded = []
    for j in range(len(features[i])):
        tmp = padarray(np.array(features[i][j]), arr_len)
        X_padded.append(tmp)
    X_padded_final.append(X_padded)
features = X_padded_final
print("=====Generating protein sequence feature=====")
print("shape features", len(features))
######################################################################
###### END of  block for only sparse encoding #################
######################################################################

X_train = []
y_train = []

for i in range(len(features)):
    f = features[i]
    for j in range(len(lab[i])):
        y_train.append(int(lab[i][j]))
        X_train.append(f[j])

print("X train size", len(X_train))
print("y train size", len(y_train))

np.save("./kmerOHE_DNA_129_xtest.npy", np.array(X_train))
np.save("./kmerOHE_DNA_129_ytest.npy", np.array(y_train))
print("DONE")
########################################################
############ End of data processing ####################
########################################################