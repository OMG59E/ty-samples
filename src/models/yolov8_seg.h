//
// Created by intellif on 23-3-31.
//

#ifndef DCL_WRAPPER_YOLOV8_SEG_H
#define DCL_WRAPPER_YOLOV8_SEG_H

#include "yolov5_seg.h"

namespace dcl {
    class YoloV8Seg : public YoloV5Seg {
    public:
        int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
        const int nm_{32};
    };
}

#endif //DCL_WRAPPER_YOLOV8_SEG_H
