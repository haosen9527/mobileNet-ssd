#include <mobileNet/pbtotensorflow.h>

#define MODEL_PATH  "./model/frozen_inference_graph.pb"
#define Image_path "./result-Img/source.jpg"
int main()
{
    pbToTensorflow test(MODEL_PATH);
    test.init();
    test.runPB(Image_path);
    return 0;
}
