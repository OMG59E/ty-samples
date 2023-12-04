//
// Created  on 22-11-10.
//
#include <cassert>
#include <cmath>

#include "yolov3_opt.h"
#include "utils/nms.h"

namespace ty {
    int YoloV3Opt::postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) {
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
            const ty::Tensor &tensor = vOutputTensors_[k];  // bs1, 3, 85, h, w
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
                        if (conf < conf_threshold_)
                            continue;

                        float w = expf(data[dn * (num_classes_ + 5) * H * W + 2 * H * W + dh * W + dw]) * anchor_sizes_[k][dn][0];
                        float h = expf(data[dn * (num_classes_ + 5) * H * W + 3 * H * W + dh * W + dw]) * anchor_sizes_[k][dn][1];

                        if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                            continue;

                        float cx = (data[dn * (num_classes_ + 5) * H * W + 0 * H * W + dh * W + dw] + dw) * strides_[k][0];
                        float cy = (data[dn * (num_classes_ + 5) * H * W + 1 * H * W + dh * W + dw] + dh) * strides_[k][1];

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
                            float score = data[dn * (num_classes_ + 5) * H * W + (5 + dc) * H * W + dh * W + dw] * conf;
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