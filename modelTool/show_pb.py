import tensorflow as tf
with tf.Session() as sess:
    with open('/home/micros/QT_pro/gitpro/catkin_new/src/data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        print graph_def
