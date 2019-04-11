#ifndef MOBILENETCONFIG_H
#define MOBILENETCONFIG_H
#include <iostream>
#include <string>
#include <vector>

struct mobileNetConfig
{
  float alpha =1.0;
  float dropout = 0.0001;
  int depth_multiplier = 1;
  bool include_top = true;
  std::string weights = "imagenet";
  int num_classes = 1000;

  //image size
  int imageSize = 300;
};


struct ssdParams
{
    int imageShape = 300;         //image size
    int numClasses = 21;         //number of Classes
    int noAnnotationLabel = 21;  //no annotation
    std::vector<std::string> featLayers = {"block4","block7","block8","block9","block10","block11"};//featLayer names
    std::vector<int> featShapes = {38,19,10,5,3,1};         //feat size
    std::vector<float> anchorSizeBound = {0.15,0.90};  //bound
    std::vector<std::vector<float> > anchorSize = {{21.0,45.0},{45.0,99.0},
                                                   {99.0,153.0},{153.0,207.0},
                                                   {207.0,261.0},{261.0,315.0}};       //init size
    std::vector<std::vector<float> > anchorRatios = {{1,0.5},{2,0.5,3,1.0/3},
                                                     {2,0.5,3,1.0/3},{2,0.5,3,1.0/3},
                                                     {2,0.5},{2,0.5}};     //w/h
    std::vector<int> anchorSteps = {8,16,32,64,100,300};
    float anchorOffset = 0.5;     //offset
    std::vector<int> normalizations = {20,-1,-1,-1,-1,-1};    //
    std::vector<float> priorScaling = {0.1,0.1,0.2,0.2};    //
};


#endif // MOBILENETCONFIG_H
