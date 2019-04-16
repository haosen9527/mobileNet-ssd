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
    int i =0;
    bool condition;
    if(i<labels.size())
    {
        condition = true;
    }
    else
    {
        condition = false;
    }

    ///
//    def body(i, feat_labels, feat_scores,                 #该函数大致意思是选择与gt box IOU最大的锚点框负责回归任务，并预测对应的边界框，如此循环
//                 feat_ymin, feat_xmin, feat_ymax, feat_xmax):
//            """Body: update feature labels, scores and bboxes.
//            Follow the original SSD paper for that purpose:
//              - assign values when jaccard > 0.5;
//              - only update if beat the score of other bboxes.
//            """
//            # Jaccard score.                                         #计算bbox与参考框的IOU值
//            label = labels[i]
//            bbox = bboxes[i]
//            jaccard = jaccard_with_anchors(bbox)
//            # Mask: check threshold + scores + no annotations + num_classes.
//            mask = tf.greater(jaccard, feat_scores)                  #当IOU大于feat_scores时，对应的mask至1，做筛选
//            # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
//            mask = tf.logical_and(mask, feat_scores > -0.5)
//            mask = tf.logical_and(mask, label < num_classes)         #label满足<21
//            imask = tf.cast(mask, tf.int64)                          #将mask转换数据类型int型
//            fmask = tf.cast(mask, dtype)                             #将mask转换数据类型float型
//            # Update values using mask.
//            feat_labels = imask * label + (1 - imask) * feat_labels  #当mask=1，则feat_labels=1；否则为0，即背景
//            feat_scores = tf.where(mask, jaccard, feat_scores)       #tf.where表示如果mask为真则jaccard，否则为feat_scores

//            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin    #选择与GT bbox IOU最大的框作为GT bbox，然后循环
//            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
//            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
//            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

//            # Check no annotation label: ignore these anchors...     #对没有标注标签的锚点框做忽视，应该是背景
//            # interscts = intersection_with_anchors(bbox)
//            # mask = tf.logical_and(interscts > ignore_threshold,
//            #                       label == no_annotation_label)
//            # # Replace scores by -1.
//            # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

//            return [i+1, feat_labels, feat_scores,
//                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]
//        # Main loop definition.
//        i = 0
//        [i, feat_labels, feat_scores,
//         feat_ymin, feat_xmin,
//         feat_ymax, feat_xmax] = tf.while_loop(condition, body,
//                                               [i, feat_labels, feat_scores,
//                                                feat_ymin, feat_xmin,
//                                                feat_ymax, feat_xmax])
//        # Transform to center / size.                               #转换为中心及长宽形式（计算补偿后的中心）
//        feat_cy = (feat_ymax + feat_ymin) / 2.  #真实预测值其实是边界框相对于先验框的转换值，encode就是为了求这个转换值
//        feat_cx = (feat_xmax + feat_xmin) / 2.
//        feat_h = feat_ymax - feat_ymin
//        feat_w = feat_xmax - feat_xmin
//        # Encode features.
//        feat_cy = (feat_cy - yref) / href / prior_scaling[0]   #(预测真实边界框中心y-参考框中心y)/参考框高/缩放尺度
//        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
//        feat_h = tf.log(feat_h / href) / prior_scaling[2]      #log(预测真实边界框高h/参考框高h)/缩放尺度
//        feat_w = tf.log(feat_w / wref) / prior_scaling[3]
//        # Use SSD ordering: x / y / w / h instead of ours.
//        feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)  #返回（cx转换值,cy转换值,w转换值,h转换值）形式的边界框的预测值（其实是预测框相对于参考框的转换）
//        return feat_labels, feat_localizations, feat_scores                         #返回目标标签，目标预测值（位置转换值），目标置信度
//        #经过我们回归得到的变换，经过变换得到真实框，所以这个地方损失函数其实是我们预测的是变换，我们实际的框和anchor之间的变换和我们预测的变换之间的loss。我们回归的是一种变换。并不是直接预测框，这个和YOLO是不一样的。和Faster RCNN是一样的

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

