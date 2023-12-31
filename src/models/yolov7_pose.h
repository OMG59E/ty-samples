//
// Created by intellif on 23-2-20.
//

#ifndef DCL_WRAPPER_YOLOV7_POSE_H
#define DCL_WRAPPER_YOLOV7_POSE_H

#include "yolov7.h"

namespace ty {
    class YoloV7Pose : public YoloV7 {
    public:
        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{1};
        const int num_keypoint_{17};
    };
}

#endif //DCL_WRAPPER_YOLOV7_POSE_H
