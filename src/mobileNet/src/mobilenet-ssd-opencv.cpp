#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const char* classNames[] = { "background",
                             "person",
                             "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                             "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                             "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                             "scissors", "teddy bear", "hair drier", "toothbrush"};

int main() {

    String weights = "./model/frozen_inference_graph.pb";
    String prototxt = "./model/ssd_mobilenet_v1_coco.pbtxt";
    dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);

    Mat frame = cv::imread("./result-Img/source.jpg");

    cv::Mat blob = cv::dnn::blobFromImage(frame,1./255,Size(frame.cols,frame.rows));
    //cout << "blob size: " << blob.size << endl;

    net.setInput(blob);
    Mat output = net.forward();
    //cout << "output size: " << output.size << endl;

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    float confidenceThreshold = 0.50;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            ostringstream ss;
            ss << confidence;
            String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

            rectangle(frame, object, Scalar(0, 255, 0),2);
            String label = String(classNames[objectClass]) + ": " + conf;
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),Size(labelSize.width, labelSize.height + baseLine)),Scalar(0, 255, 0), CV_FILLED);
            putText(frame, label, Point(xLeftBottom, yLeftBottom),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }
    imshow("image", frame);
    waitKey(0);
    return 0;
}
