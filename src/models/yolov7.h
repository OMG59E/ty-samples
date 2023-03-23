//
// Created  on 22-8-26.
//

#ifndef DCL_WRAPPER_YOLOV7_H
#define DCL_WRAPPER_YOLOV7_H

#include "yolov5.h"

namespace dcl {
    class YoloV7 : public YoloV5 {
    private:
        int min_wh_{2};
        int max_wh_{7680};
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
    };
}

#endif //DCL_WRAPPER_YOLOV7_H
