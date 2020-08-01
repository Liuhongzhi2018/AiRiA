# ckpt node visual
# import os
# from tensorflow.python import pywrap_tensorflow

# checkpoint_name = "model.ckpt"
# checkpoint_path = os.path.join('./ade20k', checkpoint_name)
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     # print(reader.get_tensor(key)) #相应的值


# pb node visual
import tensorflow as tf
import os
 
model_dir = './pretrained/ssdlite_mobilenet_v2_coco_2018_05_09/'
model_name = 'frozen_inference_graph.pb'
 
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

files = open("mv2ssd_model_node.txt","a")
create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n',file=files)
files.close()