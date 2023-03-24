//
// Created  on 22-9-13.
//

#ifndef DCL_WRAPPER_YOLOV3_H
#define DCL_WRAPPER_YOLOV3_H

#include "base/base_detection.h"

namespace dcl {
    class YoloV3 : public BaseDetector {
    public:
        /**
         *
         * @param modelPath
         * @return
         */
        int load(const std::string &modelPath) override;

        /**
         *
         * @param images
         * @return
         */
        int preprocess(const std::vector<dcl::Mat> &images) override;

        /**
         *
         * @param images
         * @param detections
         * @return
         */
        int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
        float conf_threshold_inv_{0.25f};
        const int input_sizes_[2] = {416, 416}; // wh
        const int num_classes_{80};
        const int num_per_anchors_{3};
        const int strides_[3][2] = {{32, 32}, {16, 16}, {8, 8}};
        int layer_sizes_[3][2]{};
        const float anchor_sizes_[3][3][2] = {{{116, 90}, {156, 198}, {373, 326}},
                                             {{30, 61}, {62, 45}, {59, 119}},
                                             {{10, 13}, {16, 30}, {33, 23}}};
    };
}
#endif //DCL_WRAPPER_YOLOV3_H
