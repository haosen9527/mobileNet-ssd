#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <mobileNet/mobilenetconfig.h>
#include <mobileNet/mobileNet.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/training/coordinator.h>


namespace ssd {

typedef Eigen::Tensor<float,1,Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float,4,Eigen::RowMajor> Tensor4f;

//struct about anchors params
struct anchor_struct
{
    Tensor1f anchor_height,anchor_width;
    Tensor3f anchor_x,anchor_y;
};

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class ssd :public mobileNet
{
public:
    ssd(const Scope &scope);
    ~ssd();
    void network();


public:
    /* ssd anchor*/
    vector<anchor_struct> anchors(int image_shape);
    /*Computer anchor boxes for all feature layers*/
    vector<anchor_struct> ssd_anchors_all_layers(int image_shape, vector<int> feat_shapes, vector<vector<float> > anchor_sizes, vector<vector<float> > anchor_ratios, vector<int> anchor_step, float offset);
    /*Computer SSD default anchor boxes for one feature layers*/
    anchor_struct ssd_anchor_one_layers(int image_shape, int feat_shape,
                                                 vector<float> sizes, vector<float> ratios,
                                                 int step, float offset);

    /* SSD Anchors GT ? */
    void tf_ssd_bboxes_encode(vector<int> labels, vector<float> bboxes,vector<float> anchors, int num_classes,
                              float ignore_threshold = 0.5, vector<float> prior_scaling = {0.1,0.1,0.2,0.2});
    void tf_ssd_bboxes_encode_layer();

    /* SSD losses*/
    void ssdLosses(vector<float> logits, Tensor localisations, Tensor gclasses,
                   Tensor glocalisations, vector<float> gscores, float match_threshold =0.5,
                   float negative_ratio =3.0, float alpha =1.0, float label_smoothing = 0);
    /* Smooth absolute function*/
    Output abs_smooth(Input x);

public:
    ssdParams params;
    std::vector<std::vector<float> > layers_anchors;

    struct return_result
    {
        Tensor1f oneF;
        Tensor2f twoF;
        Tensor3f threeF;
    }inline_test;
};

}//namespace ssd
