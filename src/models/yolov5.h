//
// Created  on 22-8-26.
//

#ifndef DCL_WRAPPER_YOLOV5_H
#define DCL_WRAPPER_YOLOV5_H

#include "base/base_detection.h"

namespace ty {
    class YoloV5 : public BaseDetector {
    public:

        /**
         *
         * @param images
         * @return
         */
        int preprocess(const std::vector<ty::Mat> &images) override;

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

#endif //DCL_WRAPPER_YOLOV5_H
