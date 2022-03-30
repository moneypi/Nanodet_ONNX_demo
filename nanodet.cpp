//
// Create by RangiLyu
// 2020 / 10 / 2
//

#include "nanodet.h"
#include <numeric>

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}


static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++) {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}


bool NanoDet::hasGPU = false;
NanoDet* NanoDet::detector = nullptr;

void NanoDet::getInputName() {
    size_t numInputNodes = session->GetInputCount();
    if (numInputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char *t = session->GetInputName(0, allocator);
            inputName = my_strdup(t);
            allocator.Free(t);
        }
    }
}

void NanoDet::getOutputName() {
    size_t numOutputNodes = session->GetInputCount();
    if (numOutputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char *t = session->GetOutputName(0, allocator);
            outputName = my_strdup(t);
            allocator.Free(t);
        }
    }
}

NanoDet::NanoDet(const std::string &pathStr, int numOfThread)
{
    numThread = numOfThread;
    sessionOptions.SetInterOpNumThreads(numThread);

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    #ifdef _WIN32
    std::wstring crnnPath = strToWstr(pathStr);
    session = new Ort::Session(env, crnnPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    getInputName();
    getOutputName();
}

NanoDet::~NanoDet()
{
    delete session;
    free(inputName);
    free(outputName);
}

// void NanoDet::preprocess(cv::Mat& image, ncnn::Mat& in)
// {
//     int img_w = image.cols;
//     int img_h = image.rows;

//     in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
//     //in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, this->input_width, this->input_height);


//     in.substract_mean_normalize(mean_vals, norm_vals);
// }

std::vector<float> NanoDet::preprocess(cv::Mat &src)
{
    auto inputTensorSize = src.cols * src.rows * src.channels();
    std::vector<float> inputTensorValues(inputTensorSize);
    size_t numChannels = src.channels();
    size_t imageSize = src.cols * src.rows;

    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            float data = (float) (src.data[pid * numChannels + ch] * norm_vals[ch] - mean_vals[ch] * norm_vals[ch]);
            inputTensorValues[ch * imageSize + pid] = data;
        }
    }
    return inputTensorValues;
}

std::vector<BoxInfo> NanoDet::detect(cv::Mat image, float score_threshold, float nms_threshold)
{
    std::vector<float> inputTensorValues = preprocess(image);

    std::array<int64_t, 4> inputShape{1, image.channels(), image.rows, image.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    std::vector<Ort::Value> outputTensor = session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    //double start = ncnn::get_current_time();

#if 0
    auto ex = this->Net->create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    ex.set_vulkan_compute(this->hasGPU);
#endif
    ex.input("data", input);

    ncnn::Mat out;
    ex.extract("output", out);

#endif

    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(this->input_size[0], this->input_size[1], this->strides, center_priors);

    this->decode_infer(outputTensor[0], center_priors, score_threshold, results);

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++) {
        this->nms(results[i], nms_threshold);

        for (auto box : results[i]) {
            dets.push_back(box);
        }
    }

    //double end = ncnn::get_current_time();
    //double time = end - start;
    //printf("Detect Time:%7.2f \n", time);

    return dets;
}

void NanoDet::decode_infer(Ort::Value &feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results)
{
    std::vector<int64_t> outputShape = feats.GetTensorTypeAndShapeInfo().GetShape();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());

    float *floatArray = feats.GetTensorMutableData<float>();
    // std::vector<float> outputData(floatArray, floatArray + outputCount);

    // printf("outputShape[0] = %d\n", outputShape[0]);
    // printf("outputShape[1] = %d\n", outputShape[1]);
    // printf("outputShape[2] = %d\n", outputShape[2]);

    const int num_points = center_priors.size();
    //printf("num_points:%d\n", num_points);
#if 1
    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < num_points; idx++) {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        const float* scores = floatArray + idx * outputShape[2];
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < this->num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            //std::cout << "label:" << cur_label << " score:" << score << std::endl;
            const float* bbox_pred = floatArray + idx * outputShape[2] + this->num_class;
            results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }
    }
#endif
}

BoxInfo NanoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float* dis_after_sm = new float[this->reg_max + 1];
        activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm, this->reg_max + 1);
        for (int j = 0; j < this->reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[0]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[1]);

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void NanoDet::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
