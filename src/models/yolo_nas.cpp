//
// Created by intellif on 23-5-26.
//
#include <cassert>
#include "yolo_nas.h"
#include "utils/nms.h"

namespace ty {
    int YoloNas::postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) {
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

        const ty::Tensor &box_tensor = vOutputTensors_[0];  // 1, 8400, 4
        const ty::Tensor &cls_tensor = vOutputTensors_[1];  // 1, 8400, 80

        auto* cls = (float*)(cls_tensor.data);
        auto* box = (float*)(box_tensor.data);

        detections.clear();
        const int num_anchors = box_tensor.c();
        const int step  = num_classes_;
        assert(1 == box_tensor.n());
        assert(box_tensor.d[box_tensor.nbDims-1] == num_classes_);

        for (int dn=0; dn < num_anchors; ++dn) {
            int num_cls{-1};
            float max_conf{-1};
            for (int dc = 0; dc < step; ++dc) {  // [0-80)
                float conf = cls[dn * step + dc];
                if (max_conf < conf) {
                    num_cls = dc;
                    max_conf = conf;
                }
            }
            if (max_conf < conf_threshold_)
                continue;

            // scale_coords
            int x1 = int((box[dn * 4 + 0] - pad_w) / gain);
            int y1 = int((box[dn * 4 + 1] - pad_h) / gain);
            int x2 = int((box[dn * 4 + 2] - pad_w) / gain);
            int y2 = int((box[dn * 4 + 3] - pad_h) / gain);

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
            detection.cls = num_cls;
            detection.conf = max_conf;
            detections.emplace_back(detection);
        }

        if (detections.empty())
            return 0;
        // nms
        non_max_suppression(detections, iou_threshold_);

        return 0;
    }
}