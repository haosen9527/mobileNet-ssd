#include "mobileNet/pbtotensorflow.h"

pbToTensorflow::pbToTensorflow(string pbPath)
{
    MODEL_PATH = pbPath;
    inWidth =300;
    inHeight = 300;
    classNames = { "background",
                   "person",
                   "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                   "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                   "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                   "scissors", "teddy bear", "hair drier", "toothbrush"};
}

pbToTensorflow::~pbToTensorflow()
{

}

//从文件名中读取数据
Status pbToTensorflow::ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width,
                               vector<Tensor>* out_tensors) {
    auto root = Scope::NewRootScope();
    using namespace ops;

    auto file_reader = ops::ReadFile(root.WithOpName("file_reader"),file_name);
    const int wanted_channels = 3;
    Output image_reader;
    std::size_t found = file_name.find(".png");
    if (found!=std::string::npos)
    {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,DecodePng::Channels(wanted_channels));
    }
    else if(file_name.find(".jpg")!=std::string::npos)
    {
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,DecodeJpeg::Channels(wanted_channels));
    }
    else if(file_name.find(".bmp")!=std::string::npos)
    {
        image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader,DecodeBmp::Channels(wanted_channels));
    }
    auto float_caster =Cast(root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    auto dims_expander = ExpandDims(root, float_caster, 0);
    auto resized = ResizeBilinear(root, dims_expander,Const(root.WithOpName("resize"), {input_height, input_width}));
    Transpose(root.WithOpName("transpose"),resized,{0,1,2,3});

    GraphDef graph;
    root.ToGraphDef(&graph);
    unique_ptr<Session> session(NewSession(SessionOptions()));
    TF_CHECK_OK(session->Create(graph));
    TF_CHECK_OK(session->Run({}, {"transpose"}, {}, out_tensors));//Run，获取图片数据保存到Tensor中
    std::cout<<"size:"<<out_tensors->size()<<std::endl;

    return Status::OK();
}

void pbToTensorflow::init()
{
    Status status = ReadBinaryProto(Env::Default(),MODEL_PATH,&graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
    }
    //新建session
    status = NewSession(SessionOptions(),&session);
    status = session->Create(graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
    }

    cout<<"tensorflow model load succeed"<<endl;
}


void pbToTensorflow:: runPB(string ImagePath)
{

    vector<Tensor> inputs_list;
    ReadTensorFromImageFile(ImagePath,inWidth,inHeight,&inputs_list);
    Tensor inputs = inputs_list[0];

    vector<Tensor> outputs;
    string input = "ToFloat:0";
    string test1 = "detection_boxes:0";
    string test2 = "num_detections:0";
    string output = "detection_classes:0";//graph中的输入节点和输出节点，需要预先知道/show pb
    string output1 = "detection_scores:0";

    pair<string,Tensor>img(input,inputs);
    Status status = session->Run({img},{output,output1,test1,test2}, {}, &outputs);//Run,得到运行结果，存到outputs中
    if (!status.ok()) {
        cout<<"Running model failed"<<endl;
        cout<<status.ToString()<<endl;
        return ;
    }
    //show
    Tensor boxes = Tensor(DT_FLOAT,{100,7});
    auto boxestemp = boxes.tensor<float,2>();
    for(int i=0;i<outputs[0].dim_size(1);i++)
    {
        boxestemp(i,0) = outputs[3].tensor<float,1>()(i);//num_detections
        boxestemp(i,1) = outputs[0].tensor<float,2>()(0,i);//detection_classes
        boxestemp(i,2) = outputs[1].tensor<float,2>()(0,i);//detection_scores
        boxestemp(i,3) = outputs[2].tensor<float,3>()(0,i,0);//boxes
        boxestemp(i,4) = outputs[2].tensor<float,3>()(0,i,1);
        boxestemp(i,5) = outputs[2].tensor<float,3>()(0,i,2);
        boxestemp(i,6) = outputs[2].tensor<float,3>()(0,i,3);
    }
    //use opencv show
    cv::Mat frame = cv::imread(ImagePath);
    float confidenceThreshold = 0.50;
    for (int i = 0; i < boxes.dim_size(1); i++)
    {
        float confidence = boxestemp(i, 2);
        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(boxestemp(i, 1));

            int xLeftBottom = static_cast<int>(boxestemp(i, 4) * frame.cols);
            int yLeftBottom = static_cast<int>(boxestemp(i, 3) * frame.rows);
            int xRightTop = static_cast<int>(boxestemp(i, 6) * frame.cols);
            int yRightTop = static_cast<int>(boxestemp(i, 5) * frame.rows);

            ostringstream ss;
            ss << confidence;
            cv::String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
            rectangle(frame, object, Scalar(0, 255, 0),2);
            String label = String(classNames[objectClass]) + ": " + conf;
            std::cout<<label<<std::endl;
            int baseLine = 0;
            cv::Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, cv::Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), Scalar(0, 255, 0), CV_FILLED);
            putText(frame, label, cv::Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }

    cv::imshow("image", frame);
    cv::waitKey(0);
}
