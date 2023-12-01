//
// Created by intellif on 23-6-9.
//

#include "detr.h"
#include "utils/nms.h"
#include "utils/color.h"
#include "utils/resize.h"


namespace ty {
    int Detr::preprocess(const std::vector<ty::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            ty::Mat img;
            img.data = static_cast<unsigned char *>(vInputs[n].data);
            img.phyAddr = vInputs[n].phyAddr;
            img.channels = vInputs[n].c();
            img.height = vInputs[n].h();
            img.width = vInputs[n].w();
            img.pixelFormat = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
            dclResizeCvtPaddingOp(images[n], img, LEFT_TOP, 0);
        }
        return 0;
    }

    int Detr::postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (2 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 3", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());

        const ty::Tensor &cls_tensor = vOutputTensors_[0];  // 1, 100, 92
        const ty::Tensor &box_tensor = vOutputTensors_[1];  // 1, 100, 4
        auto* cls = (float*)(cls_tensor.data);
        auto* box = (float*)(box_tensor.data);

        detections.clear();
        const int num_anchors = box_tensor.d[1];
        const int num_classes = box_tensor.d[2];
        for (int dn=0; dn < num_anchors; ++dn) {
            // step1 softmax
            float sum = 0;
            for (int dc=0; dc < num_classes; ++dc)
                sum += std::exp(cls[dn * num_anchors + dc]);

            // step2 find max confidence
            int num_cls{-1};
            float max_conf{-1};
            for (int dc = 0; dc < num_classes; ++dc) {
                float conf = std::exp(cls[dn * num_anchors + dc]) / sum;
                if (max_conf < conf) {
                    num_cls = dc;
                    max_conf = conf;
                }
            }
            if (max_conf < conf_threshold_)
                continue;

            // step3 cxcywh -> x1y1x2y2
            float cx = box[dn * num_anchors + 0];
            float cy = box[dn * num_anchors + 1];
            float w = box[dn * num_anchors + 2];
            float h = box[dn * num_anchors + 3];
            // scale_coords
            int x1 = int((cx - w * 0.5f) / gain);
            int y1 = int((cy - h * 0.5f) / gain);
            int x2 = int((cx + w * 0.5f) / gain);
            int y2 = int((cy + h * 0.5f) / gain);

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