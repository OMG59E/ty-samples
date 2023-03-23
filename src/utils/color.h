//
// Created  on 22-8-29.
//

#ifndef DCL_WRAPPER_COLOR_H
#define DCL_WRAPPER_COLOR_H

#include "dcl.h"
#include "base_type.h"

typedef enum {
    IMAGE_COLOR_BGR888_TO_BGR888_PLANAR = 0,
    IMAGE_COLOR_BGR888_TO_RGB888_PLANAR,
} colorSpace_t;

namespace dcl {
    static int bgr888ToBgr888_Planar(const dcl::Mat &src, dcl::Mat &dst) {
        const int c = src.c();
        const int h = src.h();
        const int w = src.w();
        size_t size = c * h * w;
        for (int i = 0; i < size; ++i) {
            int dc = i % c;
            int dh = i / c / w;
            int dw = i / c % w;
            dst.data[dc * h * w + dh * w + dw] = src.data[i];
        }
        dst.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PLANAR;
        return 0;
    }

    static int bgr888ToRgb888_Planar(const dcl::Mat &src, dcl::Mat &dst) {
        const int c = src.c();
        const int h = src.h();
        const int w = src.w();
        size_t size = c * h * w;
        for (int i = 0; i < size; ++i) {
            int dc = i % c;
            int dh = i / c / w;
            int dw = i / c % w;
            dst.data[(c - dc - 1) * h * w + dh * w + dw] = src.data[i];
        }
        dst.pixelFormat = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
        return 0;
    }

    static int cvtColor(const dcl::Mat &src, dcl::Mat &dst, colorSpace_t colorSpace) {
        if (dst.empty()) {
            DCL_APP_LOG(DCL_ERROR, "dst is empty");
            return -1;
        }

        switch (colorSpace) {
            case IMAGE_COLOR_BGR888_TO_BGR888_PLANAR:
                return bgr888ToBgr888_Planar(src, dst);
            case IMAGE_COLOR_BGR888_TO_RGB888_PLANAR:
                return bgr888ToRgb888_Planar(src, dst);
            default:
                return -2;
        }
    }
}

#endif //DCL_WRAPPER_COLOR_H