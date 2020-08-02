import os
import tensorflow as tf
from tensorflow.python.platform import gfile

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# # pb path root
# MODEL_DIR = '/home/zhuminchen/liuhongzhi/GitHub/pretrained/ssdlite_mobilenet_v2_coco_2018_05_09'
# model = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')

# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('visual_logs/', graph)

# eg. python model_graph.py
# tensorboard --logdir=viusal_logs --port=6006
# ssh -L 16006:127.0.0.1:6006 zhuminchen@172.10.60.151
# http://localhost:16006/


 
model = '/home/zhuminchen/liuhongzhi/GitHub/pretrained/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)