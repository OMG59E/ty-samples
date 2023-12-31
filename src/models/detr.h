//
// Created by intellif on 23-6-9.
//

#ifndef DCL_WRAPPER_DETR_H
#define DCL_WRAPPER_DETR_H

#include "base/base_detection.h"

namespace ty {
    class Detr : public BaseDetector {
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
        const int input_sizes_[2] = {800, 800}; // wh
        const int num_classes_{92};
    };
}

#endif //DCL_WRAPPER_DETR_H
