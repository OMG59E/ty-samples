//
// Created by intellif on 23-3-16.
//

#include "yolov6_face.h"
#include "utils/nms.h"

namespace dcl {
    int YoloV6Face::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (1 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 1", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 25200, 117
        auto* pred = (float*)(tensor.data);

        const int num_anchors = tensor.c();
        const int step = num_classes_ + 5 + 10;

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
            float conf = pred[dn * step + 14] * pred[dn * step + 15];
            if (conf < conf_threshold_)
                continue;

            float w = pred[dn * step + 2];
            float h = pred[dn * step + 3];

            if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                continue;

            float cx = pred[dn * step + 0];
            float cy = pred[dn * step + 1];

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
            detection.conf = conf;
            for (int k=0; k<5; ++k) {
                detection.pts[k].x = int((pred[dn * step + k * 2 + 4] - pad_w) / gain);
                detection.pts[k].y = int((pred[dn * step + k * 2 + 5] - pad_h) / gain);
            }
            detections.emplace_back(detection);
        }

        if (detections.empty())
            return 0;

        // nms
        non_max_suppression(detections, iou_threshold_);

        return 0;
    }
}