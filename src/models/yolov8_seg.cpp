//
// Created by intellif on 23-3-31.
//

#include <cassert>
#include <unistd.h>
#include "yolov8_seg.h"
#include "utils/nms.h"
#include "dcl_ive.h"


namespace ty {
    int YoloV8Seg::postprocess(const std::vector<ty::Mat> &images, std::vector<ty::detection_t> &detections) {
        if (1 != images.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_input(%d) must be equal 1", vOutputTensors_.size());
            return -1;
        }

        if (3 != vOutputTensors_.size()) {
            DCL_APP_LOG(DCL_ERROR, "num_output(%d) must be equal 3", vOutputTensors_.size());
            return -2;
        }

        float gain = (float) input_sizes_[0] / std::max(images[0].h(), images[0].w());
        float pad_h = (input_sizes_[0] - images[0].h() * gain) * 0.5f;
        float pad_w = (input_sizes_[0] - images[0].w() * gain) * 0.5f;

        const ty::Tensor &tensor = vOutputTensors_[0];  // 1, 25200, 117
        const ty::Tensor &conf_tensor = vOutputTensors_[2];  // 1 1 8400

        const int num_anchors = tensor.d[2];
        const int step = num_classes_ + 4 + nm_;

        if (1 != tensor.n()) {
            DCL_APP_LOG(DCL_ERROR, "batch size must be equal 1", vOutputTensors_.size());
            return -3;
        }

        if (tensor.c() != step) {
            DCL_APP_LOG(DCL_ERROR, "tensor.d[1](%d) must be equal step(%d)", tensor.c(), step);
            return -4;
        }

        auto* pred = (float*)(tensor.data);
        auto* conf = (int32_t*)(conf_tensor.data);
        detections.clear();
        for (int dn = 0; dn < num_anchors; ++dn) {
            int max_idx = conf[dn];
            float max_conf = pred[(max_idx + 4) * num_anchors + dn];
            if (max_conf < conf_threshold_)
                continue;

            float w = pred[2 * num_anchors + dn];
            float h = pred[3 * num_anchors + dn];

            if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                continue;

            float cx = pred[0 * num_anchors + dn];
            float cy = pred[1 * num_anchors + dn];

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
            detection.cls = max_idx;
            detection.conf = max_conf;
            for (int m=0; m<nm_; ++m)
                detection.mask[m] = pred[(4 + num_classes_ + m) * num_anchors + dn];
            detections.emplace_back(detection);
        }

        if (detections.empty())
            return 0;

        // nms
        non_max_suppression(detections, iou_threshold_);

        const ty::Tensor &protos = vOutputTensors_[1];  // 1, 32, 160, 160
        auto* proto = (float*)(protos.data);

        const int C = protos.c();
        const int H = protos.h();
        const int W = protos.w();
        const float scale = (float) W / input_sizes_[0];

        ty::Mat prob(images[0].h(), images[0].w(), DCL_PIXEL_FORMAT_YUV_400);
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
                        p += (detection.mask[dc] * proto[dc * H * W + dh * W + dw]);
                    p = 1.0f / (1.0f + expf(-p));
                    prob_.data[dh * W + dw] = round(p * 255);
                }
            }
            uint32_t chn = 0;
            dclIvePicInfo sourcePic;
            sourcePic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            sourcePic.virAddr = (uint64_t)(prob_.data);
            sourcePic.phyAddr = prob_.phyAddr;
            sourcePic.picBufferSize = prob_.size();
            sourcePic.picHeight = prob_.h();
            sourcePic.picWidth = prob_.w();
            sourcePic.picHeightStride = prob_.h();
            sourcePic.picWidthStride = prob_.w();

            dclIveCropResizeInfo transInfo;
            transInfo.dstPic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            transInfo.dstPic.virAddr = (uint64_t)(prob.data);
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
            dclError e = dcliveCropResize(chn, &sourcePic, &transInfo, count, &taskId, milliSec);
            if (e != DCL_SUCCESS) {
                DCL_APP_LOG(DCL_ERROR, "dclmpiVpcCropResize fail, error code:%d", e);
                return -1;
            }
            while (dcliveGetProcessResult(chn, taskId, milliSec) != DCL_SUCCESS) {
                usleep(1000000);
            }

            detection.prob.create(detection.box.h(), detection.box.w(), DCL_PIXEL_FORMAT_YUV_400);

            dclIveCropInfo cropInfo;
            cropInfo.dstPic.picFormat = DCL_PIXEL_FORMAT_YUV_400;
            cropInfo.dstPic.virAddr = (uint64_t)(detection.prob.data);
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

            e = dcliveCrop(chn, &(transInfo.dstPic), &cropInfo, count, &taskId, milliSec);
            if (e != DCL_SUCCESS) {
                DCL_APP_LOG(DCL_ERROR, "dclmpiVpcCrop fail, error code:%d", e);
                return -1;
            }

            while (dcliveGetProcessResult(chn, taskId, milliSec) != DCL_SUCCESS) {
                usleep(1000000);
            }
        }
        prob.free();
        return 0;
    }
}