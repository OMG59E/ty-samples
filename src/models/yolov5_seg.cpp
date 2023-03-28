//
// Created on 23-2-21.
//
#include <cassert>

#include "yolov5_seg.h"
#include "utils/nms.h"

namespace dcl {
    int YoloV5Seg::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (2 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 2", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 25200, 117

        const int num_anchors = tensor.c();
        const int step = num_classes_ + 5 + nm_;

        if (1 != tensor.n()) {
            DCL_APP_LOG(DCL_ERROR, "batch size must be equal 1", vOutputTensors_.size());
            return -3;
        }

        if (tensor.d[tensor.nbDims - 1] != step) {
            DCL_APP_LOG(DCL_ERROR, "tensor.d[tensor.nbDims-1](%d) must be equal step(%d)",
                        tensor.d[tensor.nbDims - 1], step);
            return -4;
        }

        detections.clear();
        for (int dn = 0; dn < num_anchors; ++dn) {
            float conf = tensor.data[dn * step + 4];
            if (conf < conf_threshold_)
                continue;

            float w = tensor.data[dn * step + 2];
            float h = tensor.data[dn * step + 3];

            if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                continue;

            float cx = tensor.data[dn * step + 0];
            float cy = tensor.data[dn * step + 1];

            // scale_coords
            // int x1 = int((cx - w * 0.5f - pad_w) / gain);
            // int y1 = int((cy - h * 0.5f - pad_h) / gain);
            // int x2 = int((cx + w * 0.5f - pad_w) / gain);
            // int y2 = int((cy + h * 0.5f - pad_h) / gain);

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

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
            for (int dc = 0; dc < num_classes_ + nm_; ++dc) {  // [0-80)
                tensor.data[dn * step + 5 + dc] *= conf;
                if (dc >= 0 && dc < num_classes_) {
                    float score = tensor.data[dn * step + 5 + dc];
                    if (max_conf < score) {
                        num_cls = dc;
                        max_conf = score;
                    }
                }
            }
            if (max_conf < conf_threshold_)
                continue;
            detection.cls = num_cls;
            detection.conf = max_conf;
            memcpy(detection.mask, tensor.data + dn * step + 5 + num_classes_, nm_ * sizeof(float));
            detections.emplace_back(detection);
        }

        if (detections.empty())
            return 0;

        // nms
        non_max_suppression(detections, iou_threshold_);

        const dcl::Tensor &protos = vOutputTensors_[1];  // 1, 32, 160, 160

        const int C = protos.c();
        const int H = protos.h();
        const int W = protos.w();
        const float scale = (float) W / input_sizes_[0];

        uint8_t mask[H*W];
        float prob;
        for (auto &detection: detections) {
            memset(mask, 0, H*W);
            int x1 = int(detection.box.x1 * scale);
            int y1 = int(detection.box.y1 * scale);
            int x2 = int(detection.box.x2 * scale);
            int y2 = int(detection.box.y2 * scale);

            for (int dh = y1; dh <= y2; ++dh) {
                for (int dw = x1; dw <= x2; ++dw) {
                    prob = 0;
                    for (int dc = 0; dc < C; ++dc)
                        prob += (detection.mask[dc] * protos.data[dc * H * W + dh * W + dw]);
                    prob = 1.0f / (1.0f + expf(-prob));
                    if (prob < 0.5f)
                        continue;
                    mask[dh * W + dw] = 255;
                }
            }

            cv::Mat cvMask(cv::Size(W, H), CV_8UC1, mask);
            detection.contours.clear();
            cv::findContours(cvMask, detection.contours, cv::noArray(), cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            // 映射轮廓坐标回原图
            detection.box.x1 = int((detection.box.x1 - pad_w) / gain);
            detection.box.y1 = int((detection.box.y1 - pad_h) / gain);
            detection.box.x2 = int((detection.box.x2 - pad_w) / gain);
            detection.box.y2 = int((detection.box.y2 - pad_h) / gain);
            for (auto & contour : detection.contours) {
                for (auto& pt : contour) {
                    pt.x = (pt.x / scale - pad_w) / gain;
                    pt.y = (pt.y / scale - pad_h) / gain;
                }
            }
        }

        return 0;
    }
}