#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <string>
#include <vector>

using namespace std;
using namespace tensorflow;

string MNIST_MODEL_PATH  = "/home/micros/QT_pro/gitpro/catkin_new/src/data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb";

int Read_pb()
{
    Session *session;
    Status status = NewSession(SessionOptions(),&session);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
        return 1;
    }
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(),MNIST_MODEL_PATH,&graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
        return 1;
    }
    status = session->Create(graph_def);
    if(!status.ok())
    {
        cout<<status.ToString()<<endl;
        return 1;
    }
    cout<<"tensorflow model load succeed"<<endl;

    //test
    cout<<"graph:"<<graph_def.GetTypeName()<<endl;
}

int main()
{
    Read_pb();
    return 0;
}
