import sys
sys.getdefaultencoding()
import pickle
import numpy as np
# np.set_printoptions(threshold=1000000000000000)
# path = 'E:/PR/fpnnodule_pr.pkl'

np.set_printoptions(threshold=np.inf)
path = './voc_glove_word2vec.pkl'
# path = './coco_glove_word2vec.pkl'
files = open(path,'rb')
cont = pickle.load(files,encoding='iso-8859-1')       #读取pkl文件的内容
print(cont)
print("row: ",len(cont),"col: ",len(cont[0]))
# print("row: ",len(cont['adj']),"col: ",len(cont['adj'][0]))
cont=str(cont)


obj_path = './voc_glove_word2vec_all.txt'
# obj_path = './coco_glove_word2vec.txt'
ft = open(obj_path, 'w')
ft.write(cont)
ft.close()
