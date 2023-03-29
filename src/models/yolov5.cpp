//
// Created  on 22-9-20.
//
#include <cassert>
#include <cmath>

#include "yolov5.h"
#include "utils/nms.h"
#include "utils/color.h"
#include "utils/resize.h"

namespace dcl {
    int YoloV5::preprocess(const std::vector<dcl::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            dcl::Mat img;
            img.data = static_cast<unsigned char *>(vInputs[n].data);
            img.phyaddr = vInputs[n].phyaddr;
            img.channels = vInputs[n].c();
            img.height = vInputs[n].h();
            img.width = vInputs[n].w();
            img.pixelFormat = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
            dclResizeCvtPaddingOp(images[n], img, CENTER, 114);
        }
        return 0;
    }

    int YoloV5::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (1 != vOutputTensors_.size() && 4 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 3", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 8400, 85
        detections.clear();
        const int num_anchors = tensor.c();
        const int step = num_classes_ + 5;
        assert(1 == tensor.n());
        assert(tensor.d[tensor.nbDims-1] == step);

        for (int dn=0; dn < num_anchors; ++dn) {
            float obj_conf = tensor.data[dn * step + 4];
            if (obj_conf < conf_threshold_)
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
            for (int dc = 0; dc < num_classes_; ++dc) {  // [0-80)
                float conf = tensor.data[dn * step + 5 + dc] * obj_conf;
                if (max_conf < conf) {
                    num_cls = dc;
                    max_conf = conf;
                }
            }
            if (max_conf < conf_threshold_)
                continue;
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