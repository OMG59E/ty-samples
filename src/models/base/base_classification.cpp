//
// Created  on 22-9-20.
//

#include "base_classification.h"

namespace ty {
    int BaseClassifier::load(const std::string &modelPath) {
        return net_.load(modelPath);
    }

    int BaseClassifier::preprocess(const std::vector<ty::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            ty::Mat img;
            img.data = static_cast<unsigned char *>(vInputs[n].data);
            img.phyAddr = vInputs[n].phyAddr;
            img.channels = vInputs[n].c();
            img.height = vInputs[n].h();
            img.width = vInputs[n].w();
            img.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PLANAR;
            dclResizeCvtPaddingOp(images[n], img, NONE);
        }
        return 0;
    }

    int BaseClassifier::inference(const ty::Mat &image, std::vector<classification_t> &outputs) {
        std::vector<ty::Mat> images = {image};
        return inference(images, outputs);
    }

    int BaseClassifier::inference(const std::vector<ty::Mat> &images, std::vector<classification_t> &outputs) {
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
                    tp0.count() / 1000.0f, tp1.count() / 1000.0f, tp2.count() / 1000.0f, tp3.count() / 1000.0f);;
        return 0;
    }

    int BaseClassifier::postprocess(const std::vector<ty::Mat> &images,
                                    std::vector<ty::classification_t> &classifications) {
        classifications.clear();
        for (const auto &tensor: vOutputTensors_) {
            auto* pred = (float*)(tensor.data);
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
                float conf = pred[c];
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

    int BaseClassifier::unload() { return net_.unload(); }
}
