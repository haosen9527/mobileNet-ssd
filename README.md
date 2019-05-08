## tensorflow模型预测对比总结
### 将通过以下的方法进行模型预测部分的实现
* Tensorflow c++
* Tensorflow python
* Opencv dnn
### 对比分析内容
* 图片输入说明(图片加载)
  *  Tensorflow c++ ：
     ```cpp
     tensorflow ops::ReadFile/DecodePng
     ```
  *  Tensorflow python ：
     ```python
     load_image_into_numpy_array(numpy)
     ```
  *  Opencv dnn : 
     ```cpp
     cv::imread(imagePath)
     ```
* 模型加载说明
  * Tensorflow c++ ：
    ```cpp
    Status status = ReadBinaryProto(Env::Default(),MODEL_PATH,&graph_def);
    ```
  * Tensorflow python:
    ```python
    with gfile.FastGFile(PATH_TO_PB) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name='')
    or 
    with open(PATH_TO_PB) as f:
        graph_def = tf.GraphDef()
        .......
    ```
  * opencv DNN ：
     ```cpp 
     dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);
     ```
### 时间对比：
 *  Opencvdnn ：detection time: 618.522 ms
 *  Tensorflow c++ ：detection time:699.195 ms
 *  Tensorflow python : detection time : 5916.458 ms
### 效果展示
* Tensorflow c++ <br>
 ![tensorflow-c++](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf-c%2B%2B.png)
* Tensorflow python <br>
 ![python](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf_python.png)
* Opencv dnn <br>
 ![opencv](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf-opencv.png)

