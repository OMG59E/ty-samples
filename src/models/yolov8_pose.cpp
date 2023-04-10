//
// Created by intellif on 23-4-10.
//
#include <cassert>

#include "yolov8_pose.h"
#include "utils/nms.h"

int dcl::YoloV8Pose::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
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

    const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 56, 8400
    auto* pred = (float*)(tensor.data);

    const int num_anchors = tensor.d[2];
    const int step = num_anchors;

    detections.clear();
    for (int dn=0; dn<num_anchors; ++dn) {
        float conf = pred[4 * step + dn];  // obj_conf
        if (conf < conf_threshold_)
            continue;

        float w = pred[2 * step + dn];
        float h = pred[3 * step + dn];

        if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
            continue;

        float cx = pred[0 * step + dn];
        float cy = pred[1 * step + dn];

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
        detection.conf = conf; // obj_conf * cls_conf
        for (int k=0; k<num_keypoint_; ++k) {
            detection.kpts[k].x = int((pred[(k * 3 + 5) * step + dn] - pad_w) / gain);
            detection.kpts[k].y = int((pred[(k * 3 + 6) * step + dn] - pad_h) / gain);
            detection.kpts[k].score = pred[(k * 3 + 7) * step + dn];
        }
        detections.emplace_back(detection);
    }

    // nms
    non_max_suppression(detections, iou_threshold_);
    return 0;
}