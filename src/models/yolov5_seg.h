//
// Created on 23-2-21.
//

#ifndef DCL_WRAPPER_YOLOV5_SEG_H
#define DCL_WRAPPER_YOLOV5_SEG_H

#include "yolov5.h"

namespace dcl {
    class YoloV5Seg : public YoloV5 {
    public:
        int load(const std::string &modelPath) override;

        int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) override;

        int unload() override;

    protected:
        const int proto_sizes_[2] = {160, 160}; // wh
        dcl::Mat prob_;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
        const int nm_{32};

    };
}

#endif //DCL_WRAPPER_YOLOV5_SEG_H
