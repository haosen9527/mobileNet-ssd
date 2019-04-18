#include "mobileNet/mobileNet.h"
namespace ssd {

mobileNet::mobileNet(const Scope &scope):scope(scope),session(scope),initialized(false)
{

}
mobileNet::~mobileNet()
{

}
mobileNet* mobileNet::preprocessInput()
{
  return this;
}
mobileNet* mobileNet::convBlock(Tensor inputs, int filters,
                                float alpha, std::vector<int> kernel, std::vector<int> strides)
{
  std::cout << inputs.DebugString() << std::endl;
  auto x = tensorflow::ops::Pad(scope.WithOpName("conv1_pad"),inputs,{{0,0},{1,1},{1,1},{0,0}});

  filters = int(filters * alpha);

  if(!initialized)
  {
      weight = Variable(scope,{kernel[0],kernel[1],3,filters},DT_FLOAT);
      biases = Variable(scope,{filters},DT_FLOAT);

      TF_CHECK_OK(session.Run({Assign(scope, weight, Multiply(scope, 0.01f, RandomUniform(scope, {kernel[0],kernel[1],3,filters}, DT_FLOAT))),
                                         Assign(scope, biases, Multiply(scope, 0.01f, RandomUniform(scope, {filters}, DT_FLOAT)))},nullptr));
      std::cout<<"run_conv:"<<std::endl;
      initialized = true;
  }

  auto conv_output = ops::Conv2D(scope.WithOpName("conv1"),x,weight,{1,strides[0],strides[1],1},"SAME");

  std::vector<Tensor> run_conv;
  session.Run({conv_output},&run_conv);
  std::cout<<"run_conv:"<<run_conv[0].DebugString()<<std::endl;

  this->BatchNorm(run_conv[0]);
  this->relu6(BN_output[0],"conv1_relu");
  convBlock_output = relu_output;

  return this;
}
mobileNet* mobileNet::depthwiseConvBlock(Input inputs, int pointwise_conv_filters,
                                         float alpha, int depth_multiplier, std::vector<int> strides, int block_id)
{
  pointwise_conv_filters = float(pointwise_conv_filters*alpha);

  weight = Variable(scope,{3,3,3,depth_multiplier},DT_FLOAT);
  biases = Variable(scope,{pointwise_conv_filters},DT_FLOAT);

  auto ZeroPadding2D = tensorflow::ops::Pad(scope,inputs,{{0,0},{1,1},{1,1},{0,0}});
  std::vector<Tensor> run_conv;
  session.Run({ZeroPadding2D},&run_conv);

  auto Depthwiseconv = tensorflow::ops::DepthwiseConv2dNative(scope.WithOpName("conv2D_dw/"+block_id),run_conv[0],weight,{1,strides[0],strides[1],1},"SAME");

  session.Run({Depthwiseconv},&run_conv);

  std::cout<<"Depthwiseconv:"
          <<run_conv[0].DebugString()<<std::endl;

  this->BatchNorm(run_conv[0]);
  this->relu6(BN_output[0],"relu_depth");
  run_conv.clear();
  auto conv2D = tensorflow::ops::Conv2D(scope.WithOpName("conv2D/"+block_id),relu_output,weight,{1,strides[0],strides[1],1},"SAME");
  session.Run({conv2D},&run_conv);
  this->BatchNorm(run_conv[0]);
  this->relu6(BN_output[0],"depthwiseConvBlock");

  return this;
}
mobileNet* mobileNet::relu6(Input inputs, std::string name )
{
  relu_output = tensorflow::ops::Relu6(scope.WithOpName(name),inputs);
  return this;
}
mobileNet* mobileNet::BatchNorm(Input inputs)
{
  Tensor scalev = Tensor(DT_FLOAT,{3});
  scalev.flat<float>().setValues({1,1,1});
  auto scale = tensorflow::ops::Variable(scope,{3},DT_FLOAT);
  auto assgin_scale = tensorflow::ops::Assign(scope,scale,scalev);

  Tensor offsetv = Tensor(DT_FLOAT,{3});
  offsetv.flat<float>().setValues({0,0,0});
  auto offset = tensorflow::ops::Variable(scope,{3},DT_FLOAT);
  auto assgin_offset = tensorflow::ops::Assign(scope,offset,offsetv);

  auto mean = Tensor();
  auto var = Tensor();

  TF_CHECK_OK(session.Run({assgin_scale,assgin_offset},nullptr));
  auto BN = tensorflow::ops::FusedBatchNorm(scope,inputs,scale,offset,mean,var);

  TF_CHECK_OK(session.Run({BN.y,BN.batch_mean,BN.batch_variance},&BN_output));
  std::cout<<"BN:"<<std::endl;
  std::cout<<BN_output[0].DebugString()<<std::endl;

  return this;

}
void mobileNet::network(float alpha, int depth_multiplier, float dropout, bool include_top,
                        std::string weights, int classes, std::string pooling)
{
  //image data rows/cols channels?
  //image size [128 or 160 or 192 or 224]==default size
  Tensor imageData = Tensor(DT_FLOAT,{1,300,300,3});
  imageData.flat<float>().setRandom();

  convBlock(imageData,32,alpha,{3,3},{2,2});
      depthwiseConvBlock(convBlock_output,64,alpha,depth_multiplier,{1,1,1,1},1);
      depthwiseConvBlock(relu_output,128,alpha,depth_multiplier,{1,2,2,1},2);
      depthwiseConvBlock(relu_output,128,alpha,depth_multiplier,{1,1,1,1},3);
      depthwiseConvBlock(relu_output,256,alpha,depth_multiplier,{1,2,2,1},4);
      depthwiseConvBlock(relu_output,256,alpha,depth_multiplier,{1,1,1,1},5);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,2,2,1},6);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},7);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},8);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},9);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},10);
      depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},11);
      depthwiseConvBlock(relu_output,1024,alpha,depth_multiplier,{1,2,2,1},12);
      depthwiseConvBlock(relu_output,1024,alpha,depth_multiplier,{1,1,1,1},13);
  if(include_top)
  {
    //shape
    int shape;
  }
  auto shape_tensor =Reshape(scope,tensorflow::ops::AvgPool(scope,relu_output,{1,224,224,1},{1,1,1,1},"SAME"),{1,1,int(1024*alpha)});
  //dorpout
  auto conv2D_temp = tensorflow::ops::Conv2D(scope,shape_tensor,classes,{1,1,1,1},"SAME");
  auto softMax = tensorflow::ops::Softmax(scope,conv2D_temp);
  auto reshape = tensorflow::ops::Reshape(scope,softMax,{classes,1});
  Output pool;
  if(pooling=="avg")
  {
    pool = tensorflow::ops::AvgPool(scope,reshape,{1,224,224,1},{1,1,1,1},"SAME");
  }
  else
  {
    pool = tensorflow::ops::MaxPool(scope,reshape,{1,224,224,1},{1,1,1,1},"SAME");
  }

}

}//namespace ssd
