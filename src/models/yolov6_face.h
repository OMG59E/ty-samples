//
// Created by intellif on 23-3-16.
//

#ifndef DCL_WRAPPER_YOLOV6_FACE_H
#define DCL_WRAPPER_YOLOV6_FACE_H

#include "yolov6.h"

namespace ty {
    class YoloV6Face : public YoloV6 {
    public:
        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{1};
    };
}

#endif //DCL_WRAPPER_YOLOV6_FACE_H
