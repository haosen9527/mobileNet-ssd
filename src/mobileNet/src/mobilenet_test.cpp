#include "mobileNet/mobileNet.h"
#include "mobileNet/mobilenetconfig.h"
#include "mobileNet/ssd.h"
#include <iostream>
#include <string>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>


int main()
{
    mobileNetConfig mcfg;

    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
    ClientSession session(scope);


    ssd::Tensor1f f(10);
    f.setRandom();
    std::cout<<"f"<<f<<std::endl;
    mobileNet mobile_net(scope);
    std::cout<<mcfg.alpha<<std::endl;
   // mobile_net.network(mcfg.alpha,mcfg.depth_multiplier,mcfg.dropout,mcfg.include_top,mcfg.weights,mcfg.num_classes,"SAME");

    Tensor inputs = Tensor(DT_FLOAT,{1,300,300,3});
    inputs.flat<float>().setRandom();
    mobile_net.BatchNorm(inputs);
}
