
import tensorflow as tf
import deepdish as dd
import argparse
import os
import numpy as np
import torch
from collections import OrderedDict

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}  # python 3.6+  iteritems -> items
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("--infile", type=str,
                        help="Path to the ckpt.")
    parser.add_argument("--outfile", type=str, default='',
                        help="Output file (inferred if missing).")
    args = parser.parse_args()
    if args.outfile == '':
        args.outfile = os.path.splitext(args.infile)[0] + '.h5'
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    dd.io.save(args.outfile, weights)
    weights2 = dd.io.load(args.outfile)

    # load_dict = dd.io.load(args.outfile)
    # # print("load_dist: ",load_dict)
    # new_state_dict = OrderedDict()
    # for k, v in load_dict.items():
    #     new_state_dict[k] = v
    # torch.save(new_state_dict,'mv2ssd_pt.pth')
    # checkpoint = torch.load('mv2ssd_pt.pth')
    # f = open('mv2ssd_pth_model.txt','a')
    # for k, v in checkpoint.items():
    #     print(k,'\t',v.shape,file=f)
    # # print('*'*6)
    # f.close()
    # print('*'*6,'Finish checkpoint','*'*6)

    load_dict = dd.io.load(args.outfile)
    # print("load_dist: ",load_dict)
    new_state_dict = {}
    for k, v in load_dict.items():
        new_state_dict[k] = torch.tensor(v)
    torch.save(new_state_dict,'mv2ssd_pt_1.pth')
    checkpoint = torch.load('mv2ssd_pt_1.pth')
    f = open('mv2ssd_pth_model.txt','a')
    for k, v in checkpoint.items():
        print(k,'\t',v.shape,file=f)
    # print('*'*6)
    f.close()
    print('*'*6,'Finish checkpoint','*'*6)