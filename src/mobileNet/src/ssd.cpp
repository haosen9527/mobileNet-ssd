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

void ssd::tf_ssd_bboxes_encode(vector<int> labels, vector<float> bboxes, vector<float> anchors, int num_classes, float ignore_threshold, vector<float> prior_scaling)
{
    /*
    with tf.name_scope(scope):
            target_labels = []
            target_localizations = []
            target_scores = []
            for i, anchors_layer in enumerate(anchors):
                with tf.name_scope('bboxes_encode_block_%i' % i):
                    t_labels, t_loc, t_scores = \
                        tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                                   num_classes, no_annotation_label,
                                                   ignore_threshold,
                                                   prior_scaling, dtype)
                    target_labels.append(t_labels)
                    target_localizations.append(t_loc)
                    target_scores.append(t_scores)
            ## t_labels 表示返回每个anchor对应的类别，t_loc返回的是一种变换，
            ## t_scores 每个anchor与gt对应的最大的交并比
            ## target_labels是一个list，包含每层的每个anchor对应的gt类别，
            ## target_localizations对应的是包含每一层所有anchor对应的变换
            ### target_scores 返回的是每个anchor与gt对应的最大的交并比
            return target_labels, target_localizations, target_scores*/

}
void ssd::tf_ssd_bboxes_encode_layer()
{

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

