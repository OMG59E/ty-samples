//
// Created on 23-2-21.
//
#include <cassert>
#include <unistd.h>
#include "yolov5_seg.h"
#include "utils/nms.h"
#include "dcl_mpi_vpc.h"
#include "opencv2/opencv.hpp"

namespace dcl {
    int YoloV5Seg::load(const std::string &modelPath) {
        prob_.create(proto_sizes_[1], proto_sizes_[0], DCL_PIXEL_FORMAT_YUV_400);
        return net_.load(modelPath);
    }

    int YoloV5Seg::unload() {
        prob_.free();
        return net_.unload();
    }

    int YoloV5Seg::postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) {
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

        const dcl::Tensor &tensor = vOutputTensors_[0];  // 1, 25200, 117

        const int num_anchors = tensor.c();
        const int step = num_classes_ + 5 + nm_;

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
            float conf = tensor.data[dn * step + 4];
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
            int num_cls{-1};
            float max_conf{-1};
            for (int dc = 0; dc < num_classes_ + nm_; ++dc) {  // [0-80)
                tensor.data[dn * step + 5 + dc] *= conf;
                if (dc >= 0 && dc < num_classes_) {
                    float score = tensor.data[dn * step + 5 + dc];
                    if (max_conf < score) {
                        num_cls = dc;
                        max_conf = score;
                    }
                }
            }
            if (max_conf < conf_threshold_)
                continue;
            detection.cls = num_cls;
            detection.conf = max_conf;
            memcpy(detection.mask, tensor.data + dn * step + 5 + num_classes_, nm_ * sizeof(float));
            detections.emplace_back(detection);
        }

        if (detections.empty())
            return 0;

        // nms
        non_max_suppression(detections, iou_threshold_);

        const dcl::Tensor &protos = vOutputTensors_[1];  // 1, 32, 160, 160

        const int C = protos.c();
        const int H = protos.h();
        const int W = protos.w();
        const float scale = (float) W / input_sizes_[0];

        dcl::Mat prob(images[0].h(), images[0].w(), DCL_PIXEL_FORMAT_YUV_400);
        for (auto &detection: detections) {
            memset(prob_.data, 0, prob_.size());
            memset(prob.data, 0, prob.size());
            int x1 = int((detection.box.x1 * gain + pad_w) * scale);
            int y1 = int((detection.box.y1 * gain + pad_h) * scale);
            int x2 = int((detection.box.x2 * gain + pad_w) * scale);
            int y2 = int((detection.box.y2 * gain + pad_h) * scale);

            for (int dh = y1; dh <= y2; ++dh) {
                for (int dw = x1; dw <= x2; ++dw) {
                    float p = 0;
                    for (int dc = 0; dc < C; ++dc)
                        p += (detection.mask[dc] * protos.data[dc * H * W + dh * W + dw]);
                    p = 1.0f / (1.0f + expf(-p));
                    prob_.data[dh * W + dw] = round(p * 255);
                }
            }

            uint32_t chn = 0;
            dclVpcPicInfo sourcePic;
            sourcePic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            sourcePic.picAddr = prob_.data;
            sourcePic.phyAddr = prob_.phyAddr;
            sourcePic.picBufferSize = prob_.size();
            sourcePic.picHeight = prob_.h();
            sourcePic.picWidth = prob_.w();
            sourcePic.picHeightStride = prob_.h();
            sourcePic.picWidthStride = prob_.w();

            dclVpcCropResizeInfo transInfo;
            transInfo.dstPic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            transInfo.dstPic.picAddr = prob.data;
            transInfo.dstPic.phyAddr = prob.phyAddr;
            transInfo.dstPic.picBufferSize = prob.size();
            transInfo.dstPic.picHeight = prob.h();
            transInfo.dstPic.picWidth = prob.w();
            transInfo.dstPic.picHeightStride = prob.h();
            transInfo.dstPic.picWidthStride = prob.w();
            transInfo.crop.roi.x = images[0].w() > images[0].h() ? 0 : (W - int(gain * scale * images[0].w())) / 2;
            transInfo.crop.roi.y = images[0].w() > images[0].h() ? (H - int(gain * scale * images[0].h())) / 2 : 0;
            transInfo.crop.roi.width = images[0].w() > images[0].h() ? W : int(gain * scale * images[0].w());
            transInfo.crop.roi.height = images[0].w() > images[0].h() ? int(gain * scale * images[0].h()) : H;
            transInfo.resize.width = prob.w();
            transInfo.resize.height = prob.h();
            transInfo.resize.interpolation = 0;

            uint32_t count = 1;
            uint64_t taskId;
            int32_t milliSec = -1;
            dclError e = dclmpiVpcCropResize(chn, &sourcePic, &transInfo, count, &taskId, milliSec);
            if (e != DCL_SUCCESS) {
                DCL_APP_LOG(DCL_ERROR, "dclmpiVpcCropResize fail, error code:%d", e);
                return -1;
            }
            while (dclmpiVpcGetProcessResult(chn, taskId, milliSec) != DCL_SUCCESS) {
                usleep(1000000);
            }

            detection.prob.create(detection.box.h(), detection.box.w(), DCL_PIXEL_FORMAT_YUV_400);

            dclVpcCropInfo cropInfo;
            cropInfo.dstPic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            cropInfo.dstPic.picAddr = detection.prob.data;
            cropInfo.dstPic.phyAddr = detection.prob.phyAddr;
            cropInfo.dstPic.picBufferSize = detection.prob.size();
            cropInfo.dstPic.picHeight = detection.prob.h();
            cropInfo.dstPic.picWidth = detection.prob.w();
            cropInfo.dstPic.picHeightStride = detection.prob.h();
            cropInfo.dstPic.picWidthStride = detection.prob.w();
            cropInfo.crop.roi.x = detection.box.x1;
            cropInfo.crop.roi.y = detection.box.y1;
            cropInfo.crop.roi.width = detection.box.w();
            cropInfo.crop.roi.height = detection.box.h();

            e = dclmpiVpcCrop(chn, &(transInfo.dstPic), &cropInfo, count, &taskId, milliSec);
            if (e != DCL_SUCCESS) {
                DCL_APP_LOG(DCL_ERROR, "dclmpiVpcCrop fail, error code:%d", e);
                return -1;
            }

            while (dclmpiVpcGetProcessResult(chn, taskId, milliSec) != DCL_SUCCESS) {
                usleep(1000000);
            }
        }

        prob.free();
        return 0;
    }
}