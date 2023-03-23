//
// Created on 23-2-20.
//
#include <cassert>

#include "yolov7_pose.h"
#include "utils/nms.h"

int dcl::YoloV7Pose::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
    if (1 != images.size()) {
        DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
        return -1;
    }

    if (5 != vOutputTensors_.size()) {
        DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 5", vOutputTensors_.size());
        return -2;
    }

    float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
    float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
    float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

    const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 25500, 57

    const int num_anchors = tensor.c();
    const int step = tensor.h(); // 57
    assert(1 == tensor.n());
    assert(tensor.d[tensor.nbDims-1] == step);

    detections.clear();
    for (int dn=0; dn<num_anchors; ++dn) {
        float conf = tensor.data[dn * step + 4];  // obj_conf
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
        detection.cls = 0;
        detection.conf = tensor.data[dn * step + 5] * conf; // obj_conf * cls_conf
        for (int k=0; k<num_keypoint_; ++k) {
            detection.kpts[k].x = int((tensor.data[dn * step + k * 3 + 6] - pad_w) / gain);
            detection.kpts[k].y = int((tensor.data[dn * step + k * 3 + 7] - pad_h) / gain);
            detection.kpts[k].score = tensor.data[dn * step + k * 3 + 8];
        }
        detections.emplace_back(detection);
    }

    // nms
    non_max_suppression(detections, iou_threshold_);
    return 0;
}

