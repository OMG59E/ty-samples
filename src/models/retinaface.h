//
// Created  on 22-9-13.
//

#ifndef DCL_WRAPPER_RETINAFACE_H
#define DCL_WRAPPER_RETINAFACE_H

#include "base/base_detection.h"

namespace ty {
    class RetinaFace : public BaseDetector {
    public:
        /**
         *
         * @param images
         * @return
         */
        int load(const std::string &modelPath) override;

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

        /**
         *
         * @param images
         * @return
         */
        int unload() override;

    private:
        const int input_sizes_[2] = {640, 640}; // wh
        const float min_sizes_[6] = {16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
        const float offsets_[3] = {0.5f, 0.5f, 0.5f};
        const int steps_[6] = {8, 8, 16, 16, 32, 32};
        const float variances_[2] = {0.1f, 0.2f};
        int layer_sizes_[6]{};  // wh
        int num_anchors_{0};
        float* prior_data_{nullptr};
        // bool enable_aipp_{true};
    };
}

#endif //DCL_WRAPPER_RETINAFACE_H
