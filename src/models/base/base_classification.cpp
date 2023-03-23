//
// Created  on 22-9-20.
//

#include "base_classification.h"

namespace dcl {
    int BaseClassifier::load(const std::string &modelPath, bool enableAipp) {
        return net_.load(modelPath, enableAipp);
    }

    int BaseClassifier::inference(dcl::Mat &image, std::vector<dcl::classification_t> &classifications) {
        std::vector<dcl::Mat> images = {image};
        return inference(images, classifications);
    }

    int BaseClassifier::inference(std::vector<dcl::Mat> &images, std::vector<dcl::classification_t> &classifications) {
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
        if (0 != postprocess(images, classifications)) {
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

    int BaseClassifier::unload() {
        return net_.unload();
    }

    int BaseClassifier::postprocess(const std::vector<dcl::Mat> &images,
                                    std::vector<dcl::classification_t> &classifications) {
        classifications.clear();
        for (const auto &tensor: vOutputTensors_) {
            if (2 != tensor.nbDims) {
                DCL_APP_LOG(DCL_ERROR, "Output tensor dims(%d) must be 2", tensor.nbDims);
                return -1;
            }

            if (1 != tensor.n()) {
                DCL_APP_LOG(DCL_ERROR, "batch(%d) must be equal 1", tensor.n());
                return -2;
            }

            const int num_classes = tensor.c();
            // find max
            int max_cls = -1;
            float max_conf = 0;
            for (int c = 0; c < num_classes; ++c) {
                float conf = tensor.data[c];
                if (conf > max_conf) {
                    max_cls = c;
                    max_conf = conf;
                }
            }
            classification_t classification;
            classification.conf = max_conf;
            classification.cls = max_cls;
            classifications.emplace_back(classification);
        }
        return 0;
    }
}