//
// Created by intellif on 23-3-23.
//

#ifndef DCL_WRAPPER_RESIZE_H
#define DCL_WRAPPER_RESIZE_H

#include "utils.h"
#include "color.h"
#include "dcl_mpi.h"
#include "dcl_ive.h"
#include "dcl_memory.h"
#include "base_type.h"
#include <cassert>

namespace dcl {
    typedef enum {
        NONE = 0,
        TOP_LEFT,
        CENTER
    } paddingType_t;

    static int resize(unsigned char *in_data, int channels, int in_h, int in_w,
                      unsigned char *out_data, int out_h, int out_w, colorSpace_t colorSpace,
                      paddingType_t paddingType = NONE, unsigned char paddingValue = 114) {

        float scale_x, scale_y;
        int offset_w, offset_h;
        int target_w, target_h;
        if (NONE == paddingType) {
            target_w = out_w;
            target_h = out_h;
            scale_x = float(in_w) / target_w;
            scale_y = float(in_h) / target_h;
            offset_w = 0;
            offset_h = 0;
        } else if (TOP_LEFT == paddingType) {
            assert(out_w == out_h);
            if (in_w > in_h) {
                target_w = out_w;
                scale_x = float(in_w) / target_w;
                scale_y = scale_x;  // float(in_h) / target_h;
                target_h = int(in_h / scale_y);
            } else {
                target_h = out_h;
                scale_y = float(in_h) / target_h;
                scale_x = scale_y;
                target_w = int(in_w / scale_x);
            }
            offset_w = 0;
            offset_h = 0;
            memset(out_data, paddingValue, out_w*out_h*channels);
        } else if (CENTER == paddingType) {
            assert(out_w == out_h);
            if (in_w > in_h) {
                target_w = out_w;
                scale_x = float(in_w) / target_w;
                scale_y = scale_x;
                target_h = int(in_h / scale_y);
                offset_w = 0;
                offset_h = (target_w - target_h) / 2;
            } else {
                target_h = out_h;
                scale_y = float(in_h) / target_h;
                scale_x = scale_y;
                target_w = int(in_w / scale_x);
                offset_w = (target_h - target_w) / 2;
                offset_h = 0;
            }
            memset(out_data, paddingValue, out_w*out_h*channels);
        } else {
            DCL_APP_LOG(DCL_ERROR, "Not support padding type: %d", paddingType);
            return -1;
        }

        for (int dc = 0; dc < channels; ++dc) {
            for (int dh = 0; dh < target_h; ++dh) {
                for (int dw = 0; dw < target_w; ++dw) {
                    float fx = (dw + 0.5f) * scale_x - 0.5f;
                    float fy = (dh + 0.5f) * scale_y - 0.5f;
                    int sx = int(floor(fx));
                    int sy = int(floor(fy));
                    fx -= sx;
                    fy -= sy;

                    if (sx < 0) {
                        fx = 0;
                        sx = 0;
                    }

                    if (sx >= in_w - 1) {
                        fx = 1;
                        sx = in_w - 2;
                    }

                    if (sy < 0) {
                        fy = 0;
                        sy = 0;
                    }

                    if (sy >= in_h - 1) {
                        fy = 1;
                        sy = in_h - 2;
                    }

                    float cbufx_x = 1.0f - fx;
                    float cbufx_y = fx;

                    float cbufy_x = 1.0f - fy;
                    float cbufy_y = fy;

                    float v00 = in_data[(sy + 0) * in_w * channels + (sx + 0) * channels + dc];
                    float v01 = in_data[(sy + 1) * in_w * channels + (sx + 0) * channels + dc];
                    float v10 = in_data[(sy + 0) * in_w * channels + (sx + 1) * channels + dc];
                    float v11 = in_data[(sy + 1) * in_w * channels + (sx + 1) * channels + dc];

                    float val = cbufx_x * cbufy_x * v00 + cbufx_x * cbufy_y * v01 + cbufx_y * cbufy_x * v10 +
                                cbufx_y * cbufy_y * v11;

                    if (IMAGE_COLOR_BGR888_TO_RGB888_PLANAR == colorSpace) {
                        // hwc -> chw  + BGR -> RGB
                        out_data[(2 - dc) * out_h * out_w + (dh + offset_h) * out_w + (dw + offset_w)] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_BGR888_PLANAR == colorSpace) {
                        // hwc -> chw
                        out_data[dc * out_h * out_w + (dh + offset_h) * out_w + (dw + offset_w)] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_BGR888 == colorSpace) {
                        out_data[(dh + offset_h) * out_w * channels + (dw + offset_w) * channels + dc] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_RGB888 == colorSpace) {
                        out_data[(dh + offset_h) * out_w * channels + (dw + offset_w) * channels + (2 - dc)] = round(val);
                    } else {
                        DCL_APP_LOG(DCL_ERROR, "Not support color space: %d", colorSpace);
                        return -1;
                    }
                }
            }
        }
        return 0;
    }
}

#endif //DCL_WRAPPER_RESIZE_H
