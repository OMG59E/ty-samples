//
// Created  on 22-9-13.
//

#include "retinaface.h"
#include "utils/nms.h"
#include "utils/color.h"
#include "utils/resize.h"

#include <cassert>
#include <cmath>

namespace dcl {
    int RetinaFace::load(const std::string &modelPath) {
        // init feature map size
        for (int i = 0; i < 3; ++i) {
            layer_sizes_[i * 2 + 0] = input_sizes_[0] / steps_[i * 2 + 0];  // w
            layer_sizes_[i * 2 + 1] = input_sizes_[1] / steps_[i * 2 + 1];  // h
            num_anchors_ += layer_sizes_[i * 2 + 0] * layer_sizes_[i * 2 + 1];
        }
        num_anchors_ *= 2;
        dclError ret = dclrtMalloc((void **) &prior_data_, 1 * num_anchors_ * 2 * 4 * sizeof(float),
                                   DCL_MEM_MALLOC_NORMAL_ONLY);
        assert(DCL_SUCCESS == ret);

        uint32_t offset = 0;
        for (int i = 0; i < 3; ++i) {
            int layer_w = layer_sizes_[2 * i + 0];
            int layer_h = layer_sizes_[2 * i + 1];
            for (int h = 0; h < layer_h; ++h) {
                for (int w = 0; w < layer_w; ++w) {
                    float cx = (w + offsets_[i]) * steps_[2 * i + 0] / input_sizes_[0];
                    float cy = (h + offsets_[i]) * steps_[2 * i + 1] / input_sizes_[1];
                    for (int k = 0; k < 2; ++k) {
                        float box_w = min_sizes_[2 * i + k] / input_sizes_[0];
                        float box_h = min_sizes_[2 * i + k] / input_sizes_[1];
                        prior_data_[offset * 2 * 4 + h * layer_h * 2 * 4 + w * 2 * 4 + k * 4 + 0] = cx;
                        prior_data_[offset * 2 * 4 + h * layer_h * 2 * 4 + w * 2 * 4 + k * 4 + 1] = cy;
                        prior_data_[offset * 2 * 4 + h * layer_h * 2 * 4 + w * 2 * 4 + k * 4 + 2] = box_w;
                        prior_data_[offset * 2 * 4 + h * layer_h * 2 * 4 + w * 2 * 4 + k * 4 + 3] = box_h;
                    }
                }
            }
            offset += (layer_w * layer_h);
        }

        return net_.load(modelPath);
    }

    int RetinaFace::unload() {
        DCLRT_FREE(prior_data_);
        return net_.unload();
    }

    int RetinaFace::preprocess(const std::vector<dcl::Mat> &images) {
        if (images.size() != net_.getInputNum()) {
            DCL_APP_LOG(DCL_ERROR, "images size[%d] != model input size[%d]", images.size(), net_.getInputNum());
            return -1;
        }
        std::vector<input_t>& vInputs = net_.getInputs();
        for (int n=0; n < images.size(); ++n) {
            if (!vInputs[n].hasAipp()) {  // not aipp, manual preprocess
                // resize + hwc2chw + BGR2RGB
                return resize(images[n].data, images[n].c(), images[n].h(), images[n].w(),
                              static_cast<unsigned char*>(vInputs[n].data), vInputs[n].h(), vInputs[n].w(),
                              IMAGE_COLOR_BGR888_TO_BGR888_PLANAR, TOP_LEFT, 0);
            } else { //  AIPP not support BGR2RGB
                dcl::Mat img;
                img.data = static_cast<unsigned char*>(vInputs[n].data);
                img.channels = images[n].channels;
                img.height = images[n].height;
                img.width = images[n].width;
                img.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PLANAR;
                dcl::cvtColor(images[n], img, IMAGE_COLOR_BGR888_TO_BGR888_PLANAR);  // hwc -> chw and BGR -> RGB
                // update input info with aipp
                vInputs[n].update(img.c(), img.h(), img.w(), img.pixelFormat);
            }
        }
        return 0;
    }

    int RetinaFace::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (3 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 3", vOutputTensors_.size());
            return -2;
        }

        const int height = images[0].original_height;
        const int width = images[0].original_width;
        const int target_size = std::max(height, width);

        const dcl::Tensor &loc_data = vOutputTensors_[0];
        const dcl::Tensor &conf_data = vOutputTensors_[1];
        const dcl::Tensor &pts_data = vOutputTensors_[2];

        int bs = loc_data.n();
        if (1 != bs) {
            DCL_APP_LOG(DCL_ERROR, "batch(%d) must be equal 1", bs);
            return -3;
        }

        if (loc_data.c() != num_anchors_) {
            DCL_APP_LOG(DCL_ERROR, "dim1(%d) must be equal num_anchors(%d)", loc_data.c(), num_anchors_);
            return -4;
        }

        detections.clear();
        for (int i = 0; i < num_anchors_; ++i) {
            float conf = conf_data.data[i * 2 + 1]; // face conf
            if (conf < conf_threshold_)
                continue;

            float cx = prior_data_[i * 4 + 0] + loc_data.data[i * 4 + 0] * variances_[0] * prior_data_[i * 4 + 2];
            float cy = prior_data_[i * 4 + 1] + loc_data.data[i * 4 + 1] * variances_[0] * prior_data_[i * 4 + 3];
            float w = prior_data_[i * 4 + 2] * expf(loc_data.data[i * 4 + 2] * variances_[1]);
            float h = prior_data_[i * 4 + 3] * expf(loc_data.data[i * 4 + 3] * variances_[1]);

            detection_t detection;
            detection.conf = conf;
            detection.cls = 1;  // face
            detection.name = "face";
            detection.box.x1 = int((cx - w * 0.5f) * target_size);
            detection.box.y1 = int((cy - h * 0.5f) * target_size);
            detection.box.x2 = int((cx + w * 0.5f) * target_size);
            detection.box.y2 = int((cy + h * 0.5f) * target_size);
            for (int k = 0; k < 5; ++k) {
                float px = prior_data_[i * 4 + 0] + pts_data.data[i * 10 + 2 * k + 0] * variances_[0] * prior_data_[i * 4 + 2];
                float py = prior_data_[i * 4 + 1] + pts_data.data[i * 10 + 2 * k + 1] * variances_[0] * prior_data_[i * 4 + 3];
                detection.pts[k].x = int(px * target_size);
                detection.pts[k].y = int(py * target_size);
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