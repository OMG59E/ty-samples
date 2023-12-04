//
// Created  on 22-11-10.
//

#ifndef DCL_WRAPPER_YOLOV3_OPT_H
#define DCL_WRAPPER_YOLOV3_OPT_H

#include "yolov3.h"

namespace ty {
    class YoloV3Opt : public YoloV3 {
    public:
        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};

        const int input_sizes_[2] = {416, 416}; // wh
        const int num_classes_{80};
        const int num_per_anchors_{3};
        const int strides_[3][2] = {{32, 32}, {16, 16}, {8, 8}};
        int layer_sizes_[3][2]{};
        const float anchor_sizes_[3][3][2] = {{{116, 90}, {156, 198}, {373, 326}},
                                              {{30, 61}, {62, 45}, {59, 119}},
                                              {{10, 13}, {16, 30}, {33, 23}}};
        // bool enable_aipp_{true};
    };
}

#endif //DCL_WRAPPER_YOLOV3_OPT_H
