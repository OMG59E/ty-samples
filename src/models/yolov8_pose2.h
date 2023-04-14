//
// Created by intellif on 23-4-13.
//

#ifndef DCL_WRAPPER_YOLOV8_POSE2_H
#define DCL_WRAPPER_YOLOV8_POSE2_H

#include "yolov7_pose.h"

namespace dcl {
    class YoloV8Pose2 : public YoloV7Pose {
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
        const int num_classes_{1};
        const int num_keypoint_{17};
        const float strides_[3] = {8, 16, 32};
    };
}

#endif //DCL_WRAPPER_YOLOV8_POSE2_H
