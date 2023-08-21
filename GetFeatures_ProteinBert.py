import numpy as np
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

in_file = open("peth to data .txt file with seqs and labels", 'r').readlines()
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

pretrained_model_generator, input_encoder = load_pretrained_model()
local_rep = []
global_rep = []

for i in range(range(len(seqs))):
    seq_len = len(seqs[i])
    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
    encoded_x = input_encoder.encode_X(seqs[i], seq_len)
    local_representations, global_representations = model.predict(encoded_x)
    local_rep.append(local_representations)
    global_rep.append(global_representations)
    if(i%10 ==0):
        print(i)

np.save("path to .npy file to save the generated embeddings", global_rep)
print("Done")