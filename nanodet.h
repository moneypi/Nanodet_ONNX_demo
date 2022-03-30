//
// Create by RangiLyu
// 2020 / 10 / 2
//

#ifndef NANODET_H
#define NANODET_H
#include "onnxruntime_cxx_api.h"
#include <opencv2/core/core.hpp>

#ifdef _WIN32
#define my_strtol wcstol
#define my_strrchr wcsrchr
#define my_strcasecmp _wcsicmp
#define my_strdup _strdup
#else
#define my_strtol strtol
#define my_strrchr strrchr
#define my_strcasecmp strcasecmp
#define my_strdup strdup
#endif

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct CenterPrior {
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet
{
public:
    NanoDet(const std::string &pathStr, int numOfThread);

    ~NanoDet();

    static NanoDet* detector;
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Nanodet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;
    static bool hasGPU;
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {320, 320}; // input height and width
    int num_class = 4; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

    std::vector<std::string> labels{"Car", "Pedestrian", "Cyclist", "Truck"};
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };

    char *inputName;
    char *outputName;
private:
    std::vector<float> preprocess(cv::Mat &src);
    void decode_infer(Ort::Value &feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    
    void getInputName();
    void getOutputName();
};


#endif //NANODET_H
