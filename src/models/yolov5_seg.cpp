//
// Created on 23-2-21.
//
#include <cassert>

#include "yolov5_seg.h"
#include "utils/nms.h"
#include "utils/resize.h"
#include "opencv2/opencv.hpp"

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

        uint8_t mask[16 * H * W];
        float prob[H * W];
        float p;
        const float conf_inv = -logf((1.0f / 0.5f) - 1.0f);
        for (auto &detection: detections) {
            memset(prob, 0, H*W*sizeof(float));
            memset(mask, 0, 16*H*W);
            int x1 = int((detection.box.x1 * gain + pad_w) * scale);
            int y1 = int((detection.box.y1 * gain + pad_h) * scale);
            int x2 = int((detection.box.x2 * gain + pad_w) * scale);
            int y2 = int((detection.box.y2 * gain + pad_h) * scale);

            for (int dh = y1; dh <= y2; ++dh) {
                for (int dw = x1; dw <= x2; ++dw) {
                    p = 0;
                    for (int dc = 0; dc < C; ++dc)
                        p += (detection.mask[dc] * protos.data[dc * H * W + dh * W + dw]);
                    // prob[dh * W + dw] = 1.0f / (1.0f + expf(-p));
                    prob[dh * W + dw] = p;
                }
            }

            x1 /= scale;
            y1 /= scale;
            x2 /= scale;
            y2 /= scale;

            float scale_x = scale;
            float scale_y = scale;
            for (int dh = y1; dh <= y2 ; ++dh) {
                for (int dw = x1; dw <= x2; ++dw) {
                    float fx = (dw + 0.5f) * scale_x - 0.5f;
                    float fy = (dh + 0.5f) * scale_y - 0.5f;
                    int sx = int(floor(fx));
                    int sy = int(floor(fy));
                    fx -= sx;
                    fy -= sy;

                    if (sx < 0) {
                        fx = 0;
                        sx = 0;
                    }

                    if (sx >= W - 1) {
                        fx = 1;
                        sx = W - 2;
                    }

                    if (sy < 0) {
                        fy = 0;
                        sy = 0;
                    }

                    if (sy >= H - 1) {
                        fy = 1;
                        sy = H - 2;
                    }

                    float cbufx_x = 1.0f - fx;
                    float cbufx_y = fx;

                    float cbufy_x = 1.0f - fy;
                    float cbufy_y = fy;

                    float v00 = prob[(sy + 0) * W + (sx + 0)];
                    float v01 = prob[(sy + 1) * W + (sx + 0)];
                    float v10 = prob[(sy + 0) * W + (sx + 1)];
                    float v11 = prob[(sy + 1) * W + (sx + 1)];

                    float val = cbufx_x * cbufy_x * v00 + cbufx_x * cbufy_y * v01 + cbufx_y * cbufy_x * v10 + cbufx_y * cbufy_y * v11;
                    if (val < conf_inv)
                        continue;
                    mask[dh * (4 * W) + dw] = 255;
                }
            }

            cv::Mat cvMask(cv::Size(4*W, 4*H), CV_8UC1, mask);
            detection.contours.clear();
            cv::findContours(cvMask, detection.contours, cv::noArray(), cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            // 映射轮廓坐标回原图
            for (auto & contour : detection.contours) {
                for (auto& pt : contour) {
                    pt.x = (pt.x - pad_w) / gain;
                    pt.y = (pt.y - pad_h) / gain;
                }
            }
        }

        return 0;
    }
}