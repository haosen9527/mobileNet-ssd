# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile

pwdPath = os.getcwd()
IMAGE_PATH = pwdPath+'/result-Img/source.jpg'
PATH_TO_PB = pwdPath+'/model/frozen_inference_graph.pb'
class_Names = ["background","person","bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis","snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl","banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
"toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
 "scissors", "teddy bear", "hair drier", "toothbrush"];


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def save_object_detection_result():
    sess =tf.Session()
    with gfile.FastGFile(PATH_TO_PB) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name='')
    sess.run(tf.global_variables_initializer())
    image = Image.open(IMAGE_PATH)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    #print(image_np)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = sess.graph.get_tensor_by_name('detection_scores:0')
    classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
    [boxes, scores, classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    final_score = np.squeeze(scores)
    img = cv2.imread(IMAGE_PATH)
    info = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(100):
        if scores is None or final_score[i] > 0.5:
            y_min = boxes[0][i][0]*(info[0])
            x_min = boxes[0][i][1]*(info[1])
            y_max = boxes[0][i][2]*(info[0])
            x_max = boxes[0][i][3]*(info[1])
            cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),2)
            imgzi = cv2.putText(img, (class_Names[int(classes[0][i])])+': '+str(final_score[i]), (int(x_min), int(y_min)), font, 0.5, (0, 255, 0), 0,False)

    cv2.namedWindow("Image")
    cv2.imshow("Image", imgzi)
    cv2.waitKey (0)

save_object_detection_result()
