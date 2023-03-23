//
// Created  on 22-8-25.
//

#include "base_detection.h"

namespace dcl {
    int BaseDetector::load(const std::string &modelPath, bool enableAipp) {
        return net_.load(modelPath, enableAipp);
    }

    int BaseDetector::inference(dcl::Mat &image, std::vector<dcl::detection_t> &detections) {
        std::vector<dcl::Mat> images = {image};
        return inference(images, detections);
    }

    int BaseDetector::inference(std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        if (0 != preprocess(images)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to preprocess");
            return -1;
        }
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        vOutputTensors_.clear();
        if (0 != net_.inference(images, vOutputTensors_)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to inference");
            return -2;
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        if (0 != postprocess(images, detections)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to postprocess");
            return -3;
        }
        high_resolution_clock::time_point t3 = high_resolution_clock::now();
        duration<float, std::micro> tp0 = t1 - t0;
        duration<float, std::micro> tp1 = t2 - t1;
        duration<float, std::micro> tp2 = t3 - t2;
        DCL_APP_LOG(DCL_INFO, "preprocess: %.3fms, inference: %.3fms, postprocess: %.3fms",
                    tp0.count() / 1000.0f, tp1.count() / 1000.0f, tp2.count() / 1000.0f);
        return 0;
    }

    int BaseDetector::unload() {
        return net_.unload();
    }
}