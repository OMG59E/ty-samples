//
// Created by intellif on 23-4-24.
//
#include <cassert>

#include "yolov8.h"
#include "utils/nms.h"
#include "utils/math_utils.h"

namespace ty {
    int YoloV8::load(const std::string &modelPath) {
        conf_threshold_inv_ = -logf((1.0f / conf_threshold_) - 1.0f);
        return net_.load(modelPath);
    }

    int YoloV8::postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) {
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

        float num[64];
        float den[4];
        float data[4];
        const Tensor &argmax = vOutputTensors_[1];  // [1, 1, 8400]
        const Tensor &box = vOutputTensors_[1];  // [1, 64, 8400]
        const Tensor &cls = vOutputTensors_[0];  // [1, 80, 8400]
        auto *idx_data = (int *) (argmax.data);
        auto *box_data = (float *) (box.data);  // 16 4 8400
        auto *cls_data = (float *) (cls.data);
        const int num_anchors = cls.d[2];
        for (int k = 0; k < num_anchors; ++k) {
            int max_idx = idx_data[k];
            float conf = cls_data[max_idx * num_anchors + k];
            if (conf < conf_threshold_inv_)
                continue;

            // softmax
            memset(num, 0, sizeof(float) * 64);
            memset(den, 0, sizeof(float) * 4);
            memset(data, 0, sizeof(float) * 4);
            for (int i = 0; i < 64; ++i) {
                float val = expf(box_data[i * num_anchors + k]);
                num[i] = val * (i % 16);
                den[i / 16] += val;
            }
            for (int i = 0; i < 64; ++i) {
                float val = num[i] / den[i / 16];
                data[i / 16] += val;
            }

            data[0] = k % cls.w() + 0.5f - data[0];
            data[1] = k / cls.h() + 0.5f - data[1];
            data[2] = k % cls.w() + 0.5f + data[2];
            data[3] = k / cls.h() + 0.5f + data[3];

            int stride_idx = 0;
            if (k >= 6400 && k < 8000)
                stride_idx = 1;
            if (k >= 8000)
                stride_idx = 2;
            float cx = (data[0] + data[2]) * 0.5f * strides_[stride_idx];
            float cy = (data[1] + data[3]) * 0.5f * strides_[stride_idx];
            float w = (data[2] - data[0]) * strides_[stride_idx];
            float h = (data[3] - data[1]) * strides_[stride_idx];

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
            detection.cls = 0;
            detection.conf = sigmoid(conf);
            detections.emplace_back(detection);
        }
        // nms
        non_max_suppression(detections, iou_threshold_);
        return 0;
    }
}