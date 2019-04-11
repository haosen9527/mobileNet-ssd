#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <mobileNet/mobilenetconfig.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ssd {

typedef Eigen::Tensor<float,1,Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float,4,Eigen::RowMajor> Tensor4f;

using namespace std;

class ssd
{
public:
    ssd();
    ~ssd();
    void network();


public:
    /*Computer anchor boxes for all feature layers*/
    vector<vector<float> > ssd_anchors_all_layers();
    /*Computer SSD default anchor boxes for one feature layers*/
    vector<vector<float> > ssd_anchor_one_layers(int image_shape, int feat_shape,
                                                 vector<float> sizes, vector<float> ratios,
                                                 int step, float offset);

public:
    ssdParams params;
    std::vector<std::vector<float> > layers_anchors;

    struct return_result
    {
        Tensor1f oneF;
        Tensor2f twoF;
        Tensor3f threeF;
    };
};

}//namespace ssd
