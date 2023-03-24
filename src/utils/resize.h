//
// Created by intellif on 23-3-23.
//

#ifndef DCL_WRAPPER_RESIZE_H
#define DCL_WRAPPER_RESIZE_H

#include "opencv2/opencv.hpp"
#include "utils.h"
#include "color.h"
#include "base_type.h"

namespace dcl {
    static int resize(unsigned char *in_data, int channels, int in_h, int in_w,
                      unsigned char *out_data, int out_h, int out_w, int mode) {
        float scale_x = float(in_w) / out_w;
        float scale_y = float(in_h) / out_h;
        for (int dc = 0; dc < channels; ++dc) {
            for (int dh = 0; dh < out_h; ++dh) {
                for (int dw = 0; dw < out_w; ++dw) {
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

                    if (IMAGE_COLOR_BGR888_TO_RGB888_PLANAR == mode) {
                        // hwc -> chw  + BGR -> RGB
                        out_data[(2 - dc) * out_h * out_w + dh * out_w + dw] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_BGR888_PLANAR == mode) {
                        // hwc -> chw
                        out_data[dc * out_h * out_w + dh * out_w + dw] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_BGR888 == mode) {
                        out_data[dh * out_w * channels + dw * channels + dc] = round(val);
                    } else if (IMAGE_COLOR_BGR888_TO_RGB888 == mode) {
                        out_data[dh * out_w * channels + dw * channels + (2 - dc)] = round(val);
                    } else {
                        DCL_APP_LOG(DCL_ERROR, "Not support mode: %d", mode);
                        return -1;
                    }
                }
            }
        }
        return 0;
    }
}

#endif //DCL_WRAPPER_RESIZE_H
