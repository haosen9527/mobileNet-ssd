# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf

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

    image = Image.open(IMAGE_PATH)
    image_np = load_image_into_numpy_array(image)
    #print(image_np)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    with open(PATH_TO_PB) as f:
	graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(graph_def, input_map={"image_tensor:0": image_np_expanded},
                                     return_elements=['detection_boxes:0','detection_scores:0','detection_classes:0','num_detections:0'])
    (boxes, scores, classes, num_detections) = sess.run(output)
    np.set_printoptions(threshold=1000000)
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
            imgzi = cv2.putText(img, (class_Names[int(classes[0][i])])+': '+str(final_score[i]), (int(x_min), int(y_min)), font, 0.5, (0, 255, 0), 1,True)

    cv2.namedWindow("Image")
    cv2.imshow("Image", imgzi)
    cv2.waitKey (0)
    
save_object_detection_result()
