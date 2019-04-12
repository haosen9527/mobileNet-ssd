#include "mobileNet/ssd.h"

namespace ssd {

ssd::ssd()
{

}
ssd::~ssd()
{

}

void ssd::network()
{
}

vector<anchor_struct> ssd::ssd_anchors_all_layers(int image_shape,int feat_shape,
                                                  vector<vector<float> > anchor_sizes,vector<vector<float> > anchor_ratios,
                                                  vector<int> anchor_step,float offset)
{
    vector<anchor_struct> result;
    for(int i = 0;i<feat_shape;i++)
    {
        result.push_back(ssd_anchor_one_layers(image_shape,feat_shape,anchor_sizes[i],anchor_ratios[i],anchor_step[i],offset));
    }
    return result;
}
anchor_struct ssd::ssd_anchor_one_layers(int image_shape,int feat_shape,
                                                 vector<float> sizes,vector<float> ratios,
                                                 int step,float offset)
{
    //vector<vector<int> > x,y ;
    Tensor3f x(feat_shape,feat_shape,1);
    Tensor3f y(feat_shape,feat_shape,1);
    for(int i =0;i<feat_shape;i++)
    {
        for(int j = 0;j<feat_shape;j++)
        {
            y(i,j,1) = (float)(i+offset)*(step/image_shape);
        }
    }
    for(int i =0;i<feat_shape;i++)
    {
        for(int j = 0;j<feat_shape;j++)
        {
            x(i,j,1) = (float)(j+offset)*(step/image_shape);
        }
    }
    //Expand dims to support easy broadcasting
    //extern Computer relative height and width
    int num_anchors = sizes.size()+ratios.size();
    Tensor1f h(num_anchors);
    Tensor1f w(num_anchors);
    for(int i=0;i<num_anchors;i++)
    {
        h(i) = 0;
        w(i) = 0;
    }
    //Add first anchor boxes with ratio = 1
    h(0) = sizes[0]/image_shape;
    w(0) = sizes[1]/image_shape;
    int di = 1;
    if(sizes.size()>1)
    {
        h(1) = sqrt((sizes[0]*sizes[1])/image_shape);
        w(1) = sqrt((sizes[0]*sizes[1])/image_shape);
        di++;
    }
    for(int i=0;i<ratios.size();i++)
    {
        h(i+di) = sizes[0]/image_shape/sqrt(ratios[i]);
        w(i+di) = sizes[0]/image_shape*sqrt(ratios[i]);
    }
    anchor_struct anchor_one_layer;

    x = anchor_one_layer.anchor_x;
    y = anchor_one_layer.anchor_y;
    h = anchor_one_layer.anchor_height;
    h = anchor_one_layer.anchor_width;

    return anchor_one_layer;
}

}

