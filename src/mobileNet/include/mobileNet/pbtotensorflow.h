#ifndef PBTOTENSORFLOW_H
#define PBTOTENSORFLOW_H
#include <string>
#include <vector>
#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/image_ops.h>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

//opencv
#include <opencv2/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace tensorflow;
using namespace ops;

class pbToTensorflow
{
public:
    pbToTensorflow(string pbPath);
    ~pbToTensorflow();
    Status ReadTensorFromImageFile(string file_name, const int input_height,
                                   const int input_width,
                                   vector<Tensor>* out_tensors);
    /*
     * interface API:
     *      pbToTensorflow example("*** /**.pb");
     *      example.init();
     *      ...loop...
     *      runPB("../.**.png/jpg/..")
    */
    void init();
    void runPB(string ImagePath);

private:
    //tensorflow/graph/session
    Session *session;
    GraphDef graph_def;
    //config
    string MODEL_PATH;
    size_t inWidth ;
    size_t inHeight ;
    vector<string> classNames ;
};

#endif // PBTOTENSORFLOW_H
