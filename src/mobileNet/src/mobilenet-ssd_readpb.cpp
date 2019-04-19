#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <string>
#include <vector>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace std ;
using namespace tensorflow;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

const vector<string > classNames = { "background",
                             "person",
                             "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                             "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                             "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                             "scissors", "teddy bear", "hair drier", "toothbrush"};

string MNIST_MODEL_PATH  = "/home/micros/QT_pro/gitpro/catkin_new/src/data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb";


//从文件名中读取数据
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width,
                               vector<Tensor>* out_tensors) {
    auto root = Scope::NewRootScope();
    using namespace ops;

    auto file_reader = ops::ReadFile(root.WithOpName("file_reader"),file_name);
    const int wanted_channels = 3;
    Output image_reader;
    std::size_t found = file_name.find(".png");
    //判断文件格式
    if (found!=std::string::npos) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,DecodePng::Channels(wanted_channels));
    }
    else {
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,DecodeJpeg::Channels(wanted_channels));
    }
    // 下面几步是读取图片并处理
    auto float_caster =Cast(root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    auto dims_expander = ExpandDims(root, float_caster, 0);
    auto resized = ResizeBilinear(root, dims_expander,Const(root.WithOpName("resize"), {input_height, input_width}));
    // Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),{input_std});
    Transpose(root.WithOpName("transpose"),resized,{0,2,1,3});

    GraphDef graph;
    root.ToGraphDef(&graph);

    unique_ptr<Session> session(NewSession(SessionOptions()));
    session->Create(graph);
    session->Run({}, {"transpose"}, {}, out_tensors);//Run，获取图片数据保存到Tensor中

    return Status::OK();
}


int Read_pb()
{
    GraphDef graph_def;
    Status status = ReadBinaryProto(Env::Default(),MNIST_MODEL_PATH,&graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
        return 1;
    }

    //新建session
    Session *session;
    status = NewSession(SessionOptions(),&session);
    status = session->Create(graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
        return 1;
    }

    cout<<"tensorflow model load succeed"<<endl;
    //读取图像到inputs中
    int input_height = 300;
    int input_width = 300;
    vector<Tensor> inputs;
    // string image_path(argv[1]);
    string image_path("/home/micros/catkin_ws/src/faster_rcnn_tf/data/demo/001150.jpg");
    if (!ReadTensorFromImageFile(image_path, input_height, input_width,&inputs).ok()) {
        cout<<"Read image file failed"<<endl;
        return -1;
    }

    vector<Tensor> outputs;
    string input = "ToFloat";
    string output = "detection_classes";//graph中的输入节点和输出节点，需要预先知道
    string output1 = "detection_scores";

    pair<string,Tensor>img(input,inputs[0]);
    status = session->Run({img},{output,output1}, {}, &outputs);//Run,得到运行结果，存到outputs中
    if (!status.ok()) {
        cout<<"Running model failed"<<endl;
        cout<<status.ToString()<<endl;
        return -1;
    }

    //得到模型运行结果
    Tensor t = outputs[0];
    auto tmap = t.tensor<float, 2>();
    Tensor t1 = outputs[1];
    auto tmap1 = t1.tensor<float, 2>();
    int output_dim = t.shape().dim_size(1);
    std::cout<<"detection_classes:"<<outputs[0].DebugString()
            <<endl<<output_dim<<endl
            <<endl<<tmap<<endl
           <<endl<<outputs[1].DebugString()<<endl
          <<endl<<classNames[tmap(0,1)]<<endl
            <<"detection_scores:"<<tmap1<<std::endl;


}

int main()
{
    Read_pb();
    return 0;
}
