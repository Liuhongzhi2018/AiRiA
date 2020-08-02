
#encoding:utf-8
import tensorflow as tf
import numpy as np
 
import os
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
 
# 下载下来的模型的目录
# MODEL_DIR = 'object_detection/ssd_mobilenet_v1_coco_2018_01_28'
MODEL_DIR = '/home/zhuminchen/liuhongzhi/GitHub/pretrained/ssdlite_mobilenet_v2_coco_2018_05_09'

# 下载下来的模型的文件
MODEL_CHECK_FILE = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')
# 数据集对于的label
MODEL_LABEL_MAP = os.path.join('./research/object_detection/data', 'mscoco_label_map.pbtxt')
# 数据集分类数量，可以打开mscoco_label_map.pbtxt文件看看
MODEL_NUM_CLASSES = 90
 
# 这里是获取实例图片文件名，将其放到数组中
# PATH_TO_TEST_IMAGES_DIR = './research/object_detection/test_images'
PATH_TO_TEST_IMAGES_DIR = '/home/zhuminchen/liuhongzhi/coco2017'
# TEST_IMAGES_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 6)]
TEST_IMAGES_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '000000100582.jpg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, '000000100723.jpg')]
 
# 输出图像大小，单位是in
IMAGE_SIZE = (12, 8)
 
tf.reset_default_graph()
 
# 将模型读取到默认的图中
with tf.gfile.GFile(MODEL_CHECK_FILE, 'rb') as fd:
    _graph = tf.GraphDef()
    _graph.ParseFromString(fd.read())
    tf.import_graph_def(_graph, name='')
 
# 加载COCO数据标签，将mscoco_label_map.pbtxt的内容转换成
# {1: {'id': 1, 'name': u'person'}...90: {'id': 90, 'name': u'toothbrush'}}格式
label_map = label_map_util.load_labelmap(MODEL_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MODEL_NUM_CLASSES)
category_index = label_map_util.create_category_index(categories)
 
# 将图片转化成numpy数组形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 在图中开始计算
detection_graph = tf.get_default_graph()
with tf.Session(graph=detection_graph,config=config) as sess:
    for image_path in TEST_IMAGES_PATHS:
        print(image_path)
        # 读取图片
        image = Image.open(image_path)
        # 将图片数据转成数组
        image_np = load_image_into_numpy_array(image)
        # 增加一个维度
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # 下面都是获取模型中的变量，直接使用就好了
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 存放所有检测框
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 每个检测结果的可信度
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # 每个框对应的类别
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # 检测框的个数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # 开始计算
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor : image_np_expanded})
        # 打印识别结果
        # print(num_detections)
        # print(boxes)
        # print(classes)
        # print(scores)
 
        # 得到可视化结果
        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8
        )
        # 显示
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        # plt.show()
        basename = os.path.basename(image_path)
        plt.savefig(basename)