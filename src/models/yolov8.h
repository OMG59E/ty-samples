//
// Created by intellif on 23-4-24.
//

#ifndef DCL_WRAPPER_YOLOV8_H
#define DCL_WRAPPER_YOLOV8_H

#include "yolov7.h"

namespace dcl {
    class YoloV8 : public YoloV7 {
    public:
        int load(const std::string &modelPath) override;

        int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
        float conf_threshold_inv_{0.25f};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
        const float strides_[3] = {8, 16, 32};
    };
}

#endif //DCL_WRAPPER_YOLOV8_H
