//
// Created on 23-2-21.
//

#ifndef DCL_WRAPPER_YOLOV5_SEG_H
#define DCL_WRAPPER_YOLOV5_SEG_H

#include "yolov5.h"

namespace ty {
    class YoloV5Seg : public YoloV5 {
    public:
        int load(const std::string &modelPath) override;

        int postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) override;

        int unload() override;

    protected:
        const int proto_sizes_[2] = {160, 160}; // wh
        ty::Mat prob_;

    private:
        int min_wh_{2};
        int max_wh_{7680};
        const int input_sizes_[2] = {640, 640}; // wh
        const int num_classes_{80};
        const int nm_{32};

    };
}

#endif //DCL_WRAPPER_YOLOV5_SEG_H
