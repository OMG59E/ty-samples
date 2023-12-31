//
// Created by intellif on 23-3-31.
//

#ifndef DCL_WRAPPER_YOLOV8_SEG_H
#define DCL_WRAPPER_YOLOV8_SEG_H

#include "yolov5_seg.h"

namespace ty {
    class YoloV8Seg : public YoloV5Seg {
    public:
        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
        const int nm_{32};
    };
}

#endif //DCL_WRAPPER_YOLOV8_SEG_H
