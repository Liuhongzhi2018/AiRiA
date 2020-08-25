import os
import torch
import pickle as pkl
import torch.nn as nn
import numpy as np

torch.manual_seed(2020)
embeds = nn.Embedding(35, 300)
print(embeds.weight)

save_path = './'
param_path = os.path.join(save_path, 'PETA_word2vec.pkl')
with open(param_path, 'wb') as f:
    # npembeds = np.array(embeds.weight)
    npembeds = embeds.weight.detach().numpy()
    pkl.dump(npembeds, f)

np.set_printoptions(threshold=np.inf)
path = os.path.join(save_path, 'PETA_word2vec.pkl')
# path = './coco_glove_word2vec.pkl'
files = open(path,'rb')
cont = pkl.load(files,encoding='iso-8859-1')       #读取pkl文件的内容
print("cont: ",cont)

obj_path = './PETA_word2vec.txt'
# obj_path = './coco_glove_word2vec.txt'
cont = str(cont)
ft = open(obj_path, 'w')
ft.write(cont)
ft.close()
