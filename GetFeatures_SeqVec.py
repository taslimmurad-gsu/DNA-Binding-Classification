import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import datetime
from multiprocessing import Pool

def get_embedding(sequence):
    return embedder.embed_sentence(sequence)

model_dir = Path('path of seqvecPretrain/uniref50_v2')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'

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

embedder = ElmoEmbedder(options, weights)
seqvec_spike_embed_7k = []
for i in range(len(seqs)):
    embedding = embedder.embed_sentence(seqs[i])
    residue_embd = torch.tensor(embedding).sum(dim=0)  # Tensor with shape [L,1024]
    seqvec_spike_embed_7k.append(residue_embd.detach().numpy())
    if ((i % 50) == 0):
        print("loop ", i)

print("shape seqvec training embeding", np.array(seqvec_spike_embed_7k).shape)
np.save("path to SeqVec embedding .npy", np.array(seqvec_spike_embed_7k))
print("seqvec testing embedding saved")