#include "mobileNet/ssd.h"

namespace ssd {

ssd::ssd(const Scope &scope):mobileNet(scope)
{

}
ssd::~ssd()
{

}

void ssd::network()
{

}
vector<anchor_struct> ssd:: anchors(int image_shape)
{
    return this->ssd_anchors_all_layers(image_shape,params.featShapes,params.anchorSize,params.anchorRatios,params.anchorSteps,params.anchorOffset);
}

vector<anchor_struct> ssd::ssd_anchors_all_layers(int image_shape,vector<int> feat_shapes,
                                                  vector<vector<float> > anchor_sizes,vector<vector<float> > anchor_ratios,
                                                  vector<int> anchor_step,float offset)
{
    vector<anchor_struct> result;
    for(int i = 0;i<feat_shapes.size();i++)
    {
        result.push_back(ssd_anchor_one_layers(image_shape,feat_shapes[i],anchor_sizes[i],anchor_ratios[i],anchor_step[i],offset));
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
    //test
    std::cout<<"x"<<x<<std::endl;
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
    for(int i=0;i<(ratios.size()-2);i++)
    {
        h(i+di) = sizes[0]/image_shape/sqrt(ratios[i]);
        w(i+di) = sizes[0]/image_shape*sqrt(ratios[i]);
    }
    anchor_struct anchor_one_layer;

    anchor_one_layer.anchor_x = x;
    anchor_one_layer.anchor_y = y;
    anchor_one_layer.anchor_height = h;
    anchor_one_layer.anchor_width = w;

    return anchor_one_layer;
}

vector<bboxes_struct> ssd::tf_ssd_bboxes_encode(vector<int> labels, vector<float> bboxes, vector<anchor_struct> anchors, int num_classes, float ignore_threshold, vector<float> prior_scaling)
{
    vector<bboxes_struct> target;

    for(int i=0;i<anchors.size();i++)
    {
        bboxes_struct bbox_sturct;
        bbox_sturct = tf_ssd_bboxes_encode_layer(labels,bboxes,anchors[i],num_classes,ignore_threshold,prior_scaling);
        target.push_back(bbox_sturct);
    }
    return target;
}
bboxes_struct ssd::tf_ssd_bboxes_encode_layer(vector<int> labels,vector<float> bboxes,anchor_struct anchors_layer,int num_classes,
                                              float ignore_threshold , vector<float> prior_scaling)
{
    Tensor ymin = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    Tensor ymax = Tensor(DT_FLOAT,{anchors_layer.anchor_x.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_width.dimension(0)});
    Tensor xmin = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_x.dimension(1),anchors_layer.anchor_height.dimension(0)});
    Tensor xmax = Tensor(DT_FLOAT,{anchors_layer.anchor_x.dimension(0),anchors_layer.anchor_x.dimension(1),anchors_layer.anchor_width.dimension(0)});

    auto temp_ymin = ymin.tensor<float,3>();
    auto temp_ymax = ymax.tensor<float,3>();
    auto temp_xmin = xmin.tensor<float,3>();
    auto temp_xmax = xmax.tensor<float,3>();

    for(int i = 0;i<anchors_layer.anchor_y.dimension(0);i++)
    {
        for(int j=0 ;j<anchors_layer.anchor_y.dimension(1);j++)
        {
            for(int k=0;k<anchors_layer.anchor_height.dimension(0);j++)
            {
                temp_ymin(i,j,k) = anchors_layer.anchor_y(i,j,1) - anchors_layer.anchor_height(k)/2;
                temp_ymax(i,j,k) = anchors_layer.anchor_x(i,j,1) - anchors_layer.anchor_width(k)/2;
                temp_xmin(i,j,k) = anchors_layer.anchor_y(i,j,1) + anchors_layer.anchor_height(k)/2;
                temp_xmax(i,j,k) = anchors_layer.anchor_x(i,j,1) + anchors_layer.anchor_width(k)/2;
            }
        }
    }
    //vol_anchors = (xmax - xmin) * (ymax - ymin)
    Tensor vol_anchors = Tensor(DT_FLOAT, {anchors_layer.anchor_x.dimension(0),anchors_layer.anchor_x.dimension(1),anchors_layer.anchor_width.dimension(0)});
    auto temp_vol_anchors = vol_anchors.tensor<float,3>();
    for(int i = 0;i<anchors_layer.anchor_y.dimension(0);i++)
    {
        for(int j=0 ;j<anchors_layer.anchor_y.dimension(1);j++)
        {
            for(int k=0;k<anchors_layer.anchor_height.dimension(0);j++)
            {
                temp_vol_anchors(i,j,k) = (temp_xmax(i,j,k)-temp_xmin(i,j,k)) * (temp_ymax(i,j,k)-temp_ymin(i,j,k));
            }
        }
    }
    /// Initialize tensors...
    Tensor feat_labels = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_labels.flat<float>().setZero();
    Tensor feat_scores = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_scores.flat<float>().setZero();
    Tensor feat_ymin  = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_ymin.flat<float>().setZero();
    Tensor feat_xmin = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_xmin.flat<float>().setZero();
    Tensor feat_ymax = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_ymax.flat<float>().setZero();
    Tensor feat_xmax = Tensor(DT_FLOAT,{anchors_layer.anchor_y.dimension(0),anchors_layer.anchor_y.dimension(1),anchors_layer.anchor_height.dimension(0)});
    feat_xmax.flat<float>().setZero();

    ///IOU
    auto int_ymin = ops::Maximum(scope,xmin,bboxes[0]);
    auto int_xmin = ops::Maximum(scope,ymax,bboxes[1]);
    auto int_ymax = ops::Minimum(scope,xmax,bboxes[2]);
    auto int_xmax = ops::Minimum(scope,xmax,bboxes[3]);
    auto h = ops::Maximum(scope,ops::Subtract(scope,int_ymax,int_ymin),0.0f);
    auto w = ops::Maximum(scope,ops::Subtract(scope,int_xmax,int_xmin),0.0f);
    //volumes
    auto inter_vol = ops::Multiply(scope,h,w);
    auto union_vol = ops::Add(scope,ops::Subtract(scope,vol_anchors,inter_vol),((bboxes[2]-bboxes[0])*(bboxes[3]-bboxes[1])));
    //ratios
    auto jaccard = ops::Div(scope,inter_vol,union_vol);

    ///intersection with anchors
    auto scores = ops::Div(scope,inter_vol,vol_anchors);

    ///condition
    //auto ops::Less(scope,);

}
Output ssd::abs_smooth(Input x)
{
    auto abs_x = tensorflow::ops::Abs(scope,x);
    auto minx = tensorflow::ops::Minimum(scope,abs_x,{1});
    Output r =ops::Multiply(scope,0.5f,ops::Add(scope,ops::Multiply(scope,ops::Subtract(scope,abs_x,1),minx),abs_x));

    return r;
}
void ssd::ssdLosses(vector<float> logits, Tensor localisations, Tensor gclasses, Tensor glocalisations,
                    vector<float> gscores, float match_threshold, float negative_ratio, float alpha, float label_smoothing)
{
    vector<float> temp_logit;
    vector<int> no_classes;
    //n_positives
    int n_positives =0;
    bool pmask;
    for(int i =0;i<logits.size();i++)
    {
        if(logits[i]>match_threshold)
        {
            temp_logit.push_back(1);
            no_classes.push_back(1);
            n_positives++;
            pmask = true;
        }
        else
        {
            temp_logit.push_back(0);
            no_classes.push_back(0);
            pmask = false;
        }
        // Negative mask.
        auto predictions = ops::Softmax(scope.WithOpName("block_"+i),logits[i]);
        auto nmask = ops::LogicalAnd(scope,ops::LogicalNot(scope,pmask),(gscores[i]>(-0.5)?true:false));
        auto fnmask = ops::Cast(scope,nmask,DT_FLOAT);
        auto nvalues = ops::Where(scope,predictions);

    }
}
}//namespace ssd

