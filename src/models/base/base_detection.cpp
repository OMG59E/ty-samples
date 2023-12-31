//
// Created  on 22-8-25.
//

#include "base_detection.h"

namespace ty {
    int BaseDetector::load(const std::string &modelPath) {
        return net_.load(modelPath);
    }

    int BaseDetector::preprocess(const std::vector<ty::Mat> &images) {
        return 0;
    }

    int BaseDetector::inference(const ty::Mat &image, std::vector<detection_t> &outputs) {
        std::vector<ty::Mat> images = {image};
        return inference(images, outputs);
    }

    int BaseDetector::inference(const std::vector<ty::Mat> &images, std::vector<detection_t> &outputs) {
        for (auto& image : images) {
            if (image.size() > MAX_IMAGE_SIZE) {
                DCL_APP_LOG(DCL_ERROR, "Not support image size: %d, and max support image size: %d",
                            image.size(), MAX_IMAGE_SIZE);
                return -1;
            }
        }
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        if (0 != preprocess(images)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to preprocess");
            return -1;
        }
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        vOutputTensors_.clear();
        if (0 != net_.inference(vOutputTensors_)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to inference");
            return -2;
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        if (0 != postprocess(images, outputs)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to postprocess");
            return -3;
        }
        high_resolution_clock::time_point t3 = high_resolution_clock::now();
        duration<float, std::micro> tp0 = t1 - t0;
        duration<float, std::micro> tp1 = t2 - t1;
        duration<float, std::micro> tp2 = t3 - t2;
        duration<float, std::micro> tp3 = tp0 + tp1 + tp2;
        DCL_APP_LOG(DCL_DEBUG, "preprocess: %.3fms, inference: %.3fms, postprocess: %.3fms, total: %.3fms",
                    tp0.count() / 1000.0f, tp1.count() / 1000.0f, tp2.count() / 1000.0f, tp3.count() / 1000.0f);
        return 0;
    }

    int BaseDetector::unload() { return net_.unload(); }
}