//
// Created  on 22-9-16.
//
#include <cassert>
#include <cmath>

#include "yolov3.h"
#include "utils/nms.h"
#include "utils/color.h"
#include "utils/resize.h"
#include "utils/math_utils.h"

namespace dcl {
    int YoloV3::load(const std::string &modelPath) {
        conf_threshold_inv_ = -logf((1.0f / conf_threshold_) - 1.0f);
        // init feature map size
        for (int i = 0; i < 3; ++i) {
            layer_sizes_[i][0] = input_sizes_[0] / strides_[i][0];  // w
            layer_sizes_[i][1] = input_sizes_[1] / strides_[i][1];  // h
        }
        return net_.load(modelPath);
    }

    int YoloV3::preprocess(const std::vector<dcl::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            dcl::Mat img;
            img.data = static_cast<unsigned char *>(vInputs[n].data);
            img.phyAddr = vInputs[n].phyAddr;
            img.channels = vInputs[n].c();
            img.height = vInputs[n].h();
            img.width = vInputs[n].w();
            img.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PLANAR;
            dclResizeCvtPaddingOp(images[n], img, CENTER, 128);
        }
        return 0;
    }

    int YoloV3::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (3 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 3", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        detections.clear();
        for (int k = 0; k < vOutputTensors_.size(); ++k) {
            const dcl::Tensor &tensor = vOutputTensors_[k];  // bs1, 3, 85, h, w

            auto* data = (float*)(tensor.data);

            const int C = tensor.c();
            const int H = tensor.h();
            const int W = tensor.w();

            assert(1 == tensor.n());
            assert((C / num_per_anchors_) == (num_classes_ + 5));
            assert(H == layer_sizes_[k][1]);
            assert(W == layer_sizes_[k][0]);
            for (int dh = 0; dh < H; ++dh) {
                for (int dw = 0; dw < W; ++dw) {
                    for (int dn = 0; dn < num_per_anchors_; ++dn) {  // [0-3)
                        float conf = data[dn * (num_classes_ + 5) * H * W + 4 * H * W + dh * W + dw];
                        if (conf < conf_threshold_inv_)
                            continue;

                        float w = expf(data[dn * (num_classes_ + 5) * H * W + 2 * H * W + dh * W + dw]) * anchor_sizes_[k][dn][0];
                        float h = expf(data[dn * (num_classes_ + 5) * H * W + 3 * H * W + dh * W + dw]) * anchor_sizes_[k][dn][1];

                        if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                            continue;

                        conf = sigmoid(data[dn * (num_classes_ + 5) * H * W + 4 * H * W + dh * W + dw]);

                        float cx = (sigmoid(data[dn * (num_classes_ + 5) * H * W + 0 * H * W + dh * W + dw]) + dw) * strides_[k][0];
                        float cy = (sigmoid(data[dn * (num_classes_ + 5) * H * W + 1 * H * W + dh * W + dw]) + dh) * strides_[k][1];

                        // scale_coords
                        int x1 = int((cx - w * 0.5f - pad_w) / gain);
                        int y1 = int((cy - h * 0.5f - pad_h) / gain);
                        int x2 = int((cx + w * 0.5f - pad_w) / gain);
                        int y2 = int((cy + h * 0.5f - pad_h) / gain);

                        // clip
                        x1 = x1 < 0 ? 0 : x1;
                        y1 = y1 < 0 ? 0 : y1;
                        x2 = x2 >= images[0].w() ? images[0].w() - 1 : x2;
                        y2 = y2 >= images[0].h() ? images[0].h() - 1 : y2;

                        detection_t detection;
                        detection.box.x1 = x1;
                        detection.box.y1 = y1;
                        detection.box.x2 = x2;
                        detection.box.y2 = y2;
                        int num_cls{-1};
                        float max_conf{-1};
                        for (int dc = 0; dc < num_classes_; ++dc) {  // [0-80)
                            float score = sigmoid(data[dn * (num_classes_ + 5) * H * W + (5 + dc) * H * W + dh * W + dw]) * conf;
                            if (max_conf < score) {
                                num_cls = dc;
                                max_conf = score;
                            }
                        }
                        if (max_conf < conf_threshold_)
                            continue;
                        detection.cls = num_cls;
                        detection.conf = max_conf;
                        detections.emplace_back(detection);
                    }
                }
            }
        }

        if (detections.empty())
            return 0;

        // nms
        non_max_suppression(detections, iou_threshold_);

        return 0;
    }
}