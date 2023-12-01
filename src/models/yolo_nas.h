//
// Created by intellif on 23-5-26.
//

#ifndef DCL_WRAPPER_YOLO_NAS_H
#define DCL_WRAPPER_YOLO_NAS_H

#include "yolov5.h"

namespace ty {
    class YoloNas : public YoloV5 {
    public:
        /**
         *
         * @param images
         * @param detections
         * @return
         */
        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
    };
}

#endif //DCL_WRAPPER_YOLO_NAS_H
