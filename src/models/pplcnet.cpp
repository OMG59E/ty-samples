//
// Created by intellif on 23-4-19.
//
#include "pplcnet.h"

namespace dcl {
    int PPLCNet::preprocess(const std::vector<dcl::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            if (!vInputs[n].hasAipp()) {  // not aipp, manual preprocess
                // resize + hwc2chw + BGR2RGB
                return resizeCvtPaddingOp(images[n].data, images[n].c(), images[n].h(), images[n].w(),
                                          static_cast<unsigned char*>(vInputs[n].data), vInputs[n].h(), vInputs[n].w(),
                                          IMAGE_COLOR_BGR888_TO_RGB888_PLANAR, NONE);
            } else { //  AIPP not support BGR2RGB
                dcl::Mat img;
                img.data = static_cast<unsigned char*>(vInputs[n].data);
                img.channels = images[n].channels;
                img.height = images[n].height;
                img.width = images[n].width;
                img.pixelFormat = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
                dcl::cvtColor(images[n], img, IMAGE_COLOR_BGR888_TO_RGB888_PLANAR);  // hwc -> chw and BGR -> RGB
                // update input info with aipp
                vInputs[n].update(img.c(), img.h(), img.w(), img.pixelFormat);
            }
        }
        return 0;
    }

    int PPLCNet::postprocess(const std::vector<dcl::Mat> &images, std::vector<classification_t> &outputs) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (1 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 1", vOutputTensors_.size());
            return -2;
        }

        const dcl::Tensor &tensor = vOutputTensors_[0];
        auto* data = (float*)(tensor.data);  // 1, 26
        uint8_t len = tensor.d[1];

        outputs.clear();
        outputs.resize(10);
        // gender
        outputs[0].name = data[22] > threshold_ ? "Gender: Female" : "Gender: Male";
        // age
        float max_conf = 0;
        int max_idx = -1;
        for (int k=19; k<22; ++k) {
            if (data[k] > max_conf) {
                max_conf = data[k];
                max_idx = k;
            }
        }
        outputs[1].name = "Age: " + ageList[max_idx - 19];
        // direction
        max_conf = 0;
        max_idx = -1;
        for (int k=23; k<len; ++k) {
            if (data[k] > max_conf) {
                max_conf = data[k];
                max_idx = k;
            }
        }
        outputs[2].name = "Direction: " + directList[max_idx - 23];
        // glasses
        outputs[3].name = data[1] > glasses_threshold_ ? "Glasses: True" : "Glasses: False";
        // hat
        outputs[4].name = data[0] > threshold_ ? "Hat: True" : "Hat: False";
        // hold obj
        outputs[5].name = data[18] > hold_threshold_ ? "HoldObjectsInFront: True" : "HoldObjectsInFront: False";
        // bag
        max_conf = 0;
        max_idx = -1;
        for (int k=15; k<18; ++k) {
            if (data[k] > max_conf) {
                max_conf = data[k];
                max_idx = k;
            }
        }
        outputs[6].name = max_conf > threshold_ ? "Bag: " + bagList[max_idx - 15] : "Bag: No bag";
        // upper
        outputs[7].name = "Upper: ";
        std::string sleeve = data[3] > data[2] ? "LongSleeve" : "ShortSleeve";
        outputs[7].name += sleeve;
        for (int k=4; k<8; ++k) {
            if (data[k] > threshold_) {
                outputs[7].name += " ";
                outputs[7].name += upperList[k - 4];
            }
        }
        // lower
        outputs[8].name = "Lower:";
        bool hasLower = false;
        max_idx = -1;
        for (int k=8; k<14; ++k) {
            if (data[k] > threshold_) {
                max_idx = k;
                outputs[8].name += " ";
                outputs[8].name += lowerList[k - 8];
                hasLower = true;
            }
        }
        if (!hasLower) {
            outputs[8].name += " ";
            outputs[8].name += lowerList[max_idx - 8];
        }
        // shoe
        outputs[9].name = data[14] > threshold_ ? "Shoe: Boots" : "Shoe: No boots";
        return 0;
    }
}