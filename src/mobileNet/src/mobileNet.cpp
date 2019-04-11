#include "mobileNet/mobileNet.h"


mobileNet::mobileNet(const Scope &scope):scope(scope),session(scope)
{

}
mobileNet::~mobileNet()
{

}
mobileNet* mobileNet::preprocessInput()
{
  return this;
}
mobileNet* mobileNet::convBlock(Input inputs, int filters,
                                float alpha, std::vector<int> kernel, std::vector<int> strides)
{
  auto x = tensorflow::ops::Pad(scope.WithOpName("conv1_pad"),inputs,{{1,1},{1,1}});

  filters = int(filters * alpha);
  //Tensor filter = Tensor(DT_FLOAT,{3,3,3,filters});
  //filter.flat<float>().setRandom();
  weight = Variable(scope,{kernel[0],kernel[1],3,filters},DT_FLOAT);
  biases = Variable(scope,{filters},DT_FLOAT);

  auto conv_output = ops::Conv2D(scope.WithOpName("conv1"),x,weight,{1,strides[0],strides[1],1},"SAME");
  this->BatchNorm(conv_output);
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

  auto ZeroPadding2D = tensorflow::ops::Pad(scope,inputs,{{1,1},{1,1}});
  auto Depthwiseconv = tensorflow::ops::DepthwiseConv2dNative(scope.WithOpName("conv2D_dw/"+block_id),ZeroPadding2D,weight,{1,strides[0],strides[1],1},"SAME");
  this->BatchNorm(Depthwiseconv);
  this->relu6(BN_output[0],"relu_depth");

  auto conv2D = tensorflow::ops::Conv2D(scope.WithOpName("conv2D/"+block_id),relu_output,weight,{1,strides[0],strides[1],1},"SAME");

  this->BatchNorm(conv2D);
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
  Tensor scalev = Tensor(DT_FLOAT,{4});
  scalev.flat<float>().setValues({1,1,1,1});
  auto scale = tensorflow::ops::Variable(scope,{4},DT_FLOAT);
  auto assgin_scale = tensorflow::ops::Assign(scope,scale,scalev);

  Tensor offsetv = Tensor(DT_FLOAT,{4});
  offsetv.flat<float>().setValues({0,0,0,0});
  auto offset = tensorflow::ops::Variable(scope,{4},DT_FLOAT);
  auto assgin_offset = tensorflow::ops::Assign(scope,offset,offsetv);

  auto mean = tensorflow::ops::RandomUniform(scope,{0},DT_FLOAT);
  auto var = tensorflow::ops::RandomUniform(scope,{0},DT_FLOAT);

  TF_CHECK_OK(session.Run({assgin_scale,assgin_offset},nullptr));
  auto BN = tensorflow::ops::FusedBatchNorm(scope,inputs,scale,offset,mean,var);

  TF_CHECK_OK(session.Run({BN.y,BN.batch_mean,BN.batch_variance},&BN_output));

  std::cout<<"BN:"<<std::endl;
  std::cout<<BN_output[0].DebugString()<<std::endl;

  return this;

}
void mobileNet::network(float alpha =1.0,int depth_multiplier=1,
                              float dropout =0.001,bool include_top =true,std::string weights= "imagenet",
                              int classes = 1000,std::string pooling="none")
{
  //image data rows/cols channels?
  //image size [128 or 160 or 192 or 224]==default size
  Tensor imageData = Tensor(DT_FLOAT,{1,300,300,3});
  imageData.flat<float>().setRandom();

  this->convBlock(imageData,32,alpha,{3,3},{2,2})
      ->depthwiseConvBlock(convBlock_output,64,alpha,depth_multiplier,{1,1,1,1},1)
      ->depthwiseConvBlock(relu_output,128,alpha,depth_multiplier,{1,2,2,1},2)
      ->depthwiseConvBlock(relu_output,128,alpha,depth_multiplier,{1,1,1,1},3)
      ->depthwiseConvBlock(relu_output,256,alpha,depth_multiplier,{1,2,2,1},4)
      ->depthwiseConvBlock(relu_output,256,alpha,depth_multiplier,{1,1,1,1},5)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,2,2,1},6)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},7)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},8)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},9)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},10)
      ->depthwiseConvBlock(relu_output,512,alpha,depth_multiplier,{1,1,1,1},11)
      ->depthwiseConvBlock(relu_output,1024,alpha,depth_multiplier,{1,2,2,1},12)
      ->depthwiseConvBlock(relu_output,1024,alpha,depth_multiplier,{1,1,1,1},13);
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

