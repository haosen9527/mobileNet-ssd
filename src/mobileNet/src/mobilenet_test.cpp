#include "mobileNet/mobileNet.h"
#include "mobileNet/mobilenetconfig.h"
#include "mobileNet/ssd.h"
#include <iostream>
#include <string>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
int main()
{
    mobileNetConfig mcfg;

    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(scope);


    ssd::Tensor1f f(10);
    f.setRandom();
    std::cout<<"f"<<f<<std::endl;
    ssd::mobileNet mobile_net(scope);
    std::cout<<mcfg.alpha<<std::endl;
   // mobile_net.network(mcfg.alpha,mcfg.depth_multiplier,mcfg.dropout,mcfg.include_top,mcfg.weights,mcfg.num_classes,"SAME");

    Tensor inputs = Tensor(DT_FLOAT,{1,300,300,3});
    inputs.flat<float>().setRandom();
    mobile_net.BatchNorm(inputs);
    mobile_net.convBlock(inputs,3,1);
    mobile_net.network(1);

    ssd::ssd test(scope);
    ssd::Tensor1f test1f(20);
    for(int i=0;i<20;i++)
    {
        test1f(i) = i;
    }
    test.inline_test.oneF = test1f;
    //std::cout<<test.inline_test.oneF<<std::endl;

    std::vector<std::vector<int> > test_vector = {{1,2,3},{4,5,6},{7,8}};
    std::vector<int> test_inline;
    for (int i=0;i<test_vector.size();i++)
    {
        test_inline=test_vector[i];
    }
    for(int j=0;j<test_inline.size();j++)
    {
       // std::cout<<test_inline[j];
    }

}
