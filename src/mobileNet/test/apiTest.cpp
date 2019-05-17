#include <mobileNet/pbtotensorflow.h>

#define MODEL_PATH  "/home/ywx/mobileNet-ssd/model/frozen_inference_graph.pb"
#define Image_path "/home/ywx/boxes/micros/src/micros/behavior_management/orient/data/simulation/2.jpg"
int main()
{
    pbToTensorflow test(MODEL_PATH);
    test.init();
    test.runPB(Image_path);
    return 0;
}
