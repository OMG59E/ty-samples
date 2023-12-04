//
// Created  on 22-9-21.
//

#ifndef DCL_WRAPPER_IMAGE_H
#define DCL_WRAPPER_IMAGE_H

#include <cmath>
#include <cassert>

#include "base_type.h"
#include "line.h"

namespace ty {

    void Line_VH(ty::Mat &src, ty::Point p1, ty::Point p2, ty::Color color, int thickness) {
        assert(thickness >= 1);
        assert(p2.x >= p1.x && p2.y >= p1.x);
        const int h = src.h();
        const int w = src.w();
        int pad0 = (thickness - 1) / 2;  // outside
        int pad1 = (thickness - 1) - pad0;  // inside
        int p1_x1 = p1.x - pad0;
        int p1_y1 = p1.y - pad0;
        int p2_x1 = p2.x + pad1;
        int p2_y1 = p2.y + pad1;
        // clip
        p1_x1 = p1_x1 < 0 ? 0 : p1_x1;
        p1_y1 = p1_y1 < 0 ? 0 : p1_y1;
        p2_x1 = p2_x1 >= w ? w - 1 : p2_x1;
        p2_y1 = p2_y1 >= h ? h - 1 : p2_y1;

        for (int dh=p1_y1; dh<=p2_y1; ++dh) {
            for (int dw=p1_x1; dw<=p2_x1; ++dw) {
                src.data[dh * w * 3 + dw * 3 + 0] = color.b;
                src.data[dh * w * 3 + dw * 3 + 1] = color.g;
                src.data[dh * w * 3 + dw * 3 + 2] = color.r;
            }
        }
    }

    void rectangle(ty::Mat &src, ty::Point p1, ty::Point p2, ty::Color color, int thickness) {
        Line_VH(src, p1, ty::Point(p1.x, p2.y), color, thickness);
        Line_VH(src, p1, ty::Point(p2.x, p1.y), color, thickness);
        Line_VH(src, ty::Point(p1.x, p2.y), p2, color, thickness);
        Line_VH(src, ty::Point(p2.x, p1.y), p2, color, thickness);
    }

    void circle(ty::Mat& src, ty::Point p, int radius, ty::Color color, int thickness) {
        assert(radius >= 1);
        const int h = src.h();
        const int w = src.w();
        int x1 = p.x - (radius - 1);
        int y1 = p.y - (radius - 1);
        int x2 = p.x + (radius - 1);
        int y2 = p.y + (radius - 1);
        // clip
        x1 = x1 < 0 ? 0 : x1;
        y1 = y1 < 0 ? 0 : y1;
        x2 = x2 >= w ? w - 1 : x2;
        y2 = y2 >= h ? h - 1 : y2;

        if (thickness < 0) {  // fill
            for (int dh=y1; dh<=y2; ++dh) {
                for (int dw=x1; dw<=x2; ++dw) {
                    int dx = abs(p.x - dw) + 1;
                    int dy = abs(p.y - dh) + 1;
                    int dist = dx*dx + dy*dy;
                    if (dist <= radius*radius) {
                        src.data[dh * w * 3 + dw * 3 + 0] = color.b;
                        src.data[dh * w * 3 + dw * 3 + 1] = color.g;
                        src.data[dh * w * 3 + dw * 3 + 2] = color.r;
                    }
                }
            }
        } else {
            // TODO
        }
    }
}
#endif //DCL_WRAPPER_IMAGE_H
