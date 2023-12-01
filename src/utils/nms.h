//
// Created  on 22-9-15.
//

#ifndef DCL_WRAPPER_NMS_H
#define DCL_WRAPPER_NMS_H

#include "base_type.h"
#include <vector>
#include <algorithm>


inline float bbox_overlap(const ty::Box &vi, const ty::Box &vo) {
    int xx1 = std::max(vi.x1, vo.x1);
    int yy1 = std::max(vi.y1, vo.y1);
    int xx2 = std::min(vi.x2, vo.x2);
    int yy2 = std::min(vi.y2, vo.y2);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);

    int area = w * h;

    float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                     (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);

    return dist;
}


static int non_max_suppression2(std::vector<ty::detection_t> &detections, const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const ty::detection_t &d1, const ty::detection_t &d2) { return d1.conf > d2.conf; });

    // nms
    std::vector<ty::detection_t> keep_detections;
    std::vector<ty::detection_t> tmp_detections;
    keep_detections.clear();
    while (!detections.empty()) {
        if (detections.size() == 1) {
            keep_detections.emplace_back(detections[0]);
            break;
        }

        keep_detections.emplace_back(detections[0]);

        tmp_detections.clear();
        for (int idx = 1; idx < detections.size(); ++idx) {
            float iou = bbox_overlap(keep_detections.back().box, detections[idx].box);
            if (iou < iou_threshold)
                tmp_detections.emplace_back(detections[idx]);
        }
        detections.swap(tmp_detections);
    }
    detections.swap(keep_detections);
    return 0;
}

static int non_max_suppression(std::vector<ty::detection_t> &detections, const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const ty::detection_t &d1, const ty::detection_t &d2) { return d1.conf > d2.conf; });

    // nms
    std::vector<ty::detection_t> keep_detections;
    bool *suppressed = new bool[detections.size()];
    memset(suppressed, 0, sizeof(bool) * detections.size());
    const int num_detections = detections.size();
    for (int i = 0; i < num_detections; ++i) {
        if (suppressed[i])
            continue;
        keep_detections.emplace_back(detections[i]);
        for (int j = i + 1; j < num_detections; ++j) {
            if (suppressed[j])
                continue;
            float iou = bbox_overlap(detections[i].box, detections[j].box);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
    keep_detections.swap(detections);
    delete[]suppressed;

    return 0;
}

#endif //DCL_WRAPPER_NMS_H
