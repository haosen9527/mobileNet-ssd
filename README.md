mobileNet-ssd Project

# Welcome to the mobileNet-ssd wiki!
## tensorflow模型预测对比总结
### 将通过以下的方法进行模型预测部分的实现
* Tensorflow c++
* Tensorflow python
* Opencv dnn
### 对比分析内容
* 图片输入说明(图片加载)
* * Tensorflow c++ ：使用tensorflow ops::ReadFile/DecodePng
* * Tensorflow python ：使用load_image_into_numpy_array(numpy)
* * Opencv dnn : 使用opencv imread
### 时间对比：
* * Opencvdnn ：detection time: 618.522 ms
* * Tensorflow c++ ：detection time:699.195 ms
* * Tensorflow python : detection time : 5916.458 ms
### 效果展示
* Tensorflow c++
 ![tensorflow-c++](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf-c%2B%2B.png)
* Tensorflow python
 ![python](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf_python.png)
* Opencv dnn
 ![opencv](https://github.com/haosen9527/mobileNet-ssd/blob/master/result-Img/tf-opencv.png)

