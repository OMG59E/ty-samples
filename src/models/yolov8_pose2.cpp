//
// Created by intellif on 23-4-13.
//

#include "yolov8_pose2.h"
#include "utils/nms.h"
#include "utils/math_utils.h"

namespace dcl {
    int YoloV8Pose2::load(const std::string &modelPath) {
        conf_threshold_inv_ = -logf((1.0f / conf_threshold_) - 1.0f);
        return net_.load(modelPath);
    }

    int YoloV8Pose2::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (9 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 1", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        float num[64];
        float den[4];
        float data[4];
        for (int n=0; n<3; ++n) {
            const Tensor& kpt = vOutputTensors_[0 + n];      // 0:[1,51,80,80] 1:[1,51,40,40] 2:[1,51,20,20]
            const Tensor& box = vOutputTensors_[3 + 2 * n];  // 3:[1,64,80,80] 5:[1,64,40,40] 7:[1,64,20,20]
            const Tensor& cls = vOutputTensors_[4 + 2 * n];  // 4:[1,1,80,80]  6:[1,1,40,40]  8:[1,1,20,20]
            auto* kpt_data = (float*)(kpt.data);
            auto* box_data = (float*)(box.data);  // 16 4 6400
            auto* cls_data = (float*)(cls.data);
            const int num_anchors = cls.h() * cls.w();
            for (int k=0; k<num_anchors; ++k) {
                float conf = cls_data[k];
                if (conf < conf_threshold_inv_)
                    continue;

                // softmax
                memset(num, 0, sizeof(float)*64);
                memset(den, 0, sizeof(float)*4);
                memset(data, 0, sizeof(float)*4);
                for (int i=0; i<64; ++i) {
                    float val = expf(box_data[i * num_anchors + k]);
                    num[i] = val * (i % 16);
                    den[i / 16] += val;
                }
                for (int i=0; i<64; ++i) {
                    float val = num[i] / den[i / 16];
                    data[i / 16] += val;
                }

                data[0] = k % cls.w() + 0.5f - data[0];
                data[1] = k / cls.h() + 0.5f - data[1];
                data[2] = k % cls.w() + 0.5f + data[2];
                data[3] = k / cls.h() + 0.5f + data[3];
                float cx = (data[0] + data[2]) * 0.5f * strides_[n];
                float cy = (data[1] + data[3]) * 0.5f * strides_[n];
                float w = (data[2] - data[0]) * strides_[n];
                float h = (data[3] - data[1]) * strides_[n];

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
                for (int p=0; p<num_keypoint_; ++p) {
                    detection.kpts[p].x = int(((kpt_data[(p * 3 + 0) * num_anchors + k] * 2 + k % cls.w()) * strides_[n] - pad_w) / gain);
                    detection.kpts[p].y = int(((kpt_data[(p * 3 + 1) * num_anchors + k] * 2 + k / cls.h()) * strides_[n] - pad_h) / gain);
                    detection.kpts[p].score = sigmoid(kpt_data[(p * 3 + 2) * num_anchors + k]);
                }
                detections.emplace_back(detection);
            }
        }
        // nms
        non_max_suppression(detections, iou_threshold_);
        return 0;
    }
}