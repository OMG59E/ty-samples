//
// Created on 23-2-21.
//

#ifndef DCL_WRAPPER_LINE_H
#define DCL_WRAPPER_LINE_H

#include <cmath>
#include "base_type.h"

#define fpart(x) (float(x) - int(x))
#define rfpart(x) (1.0f - fpart(x))


namespace dcl {
    static inline void setPixel(const dcl::Mat &src, int x, int y, const dcl::Color &color) {
        const int W = src.w();
        const int C = src.c();
        src.data[y * W * C + x * C + 0] = (unsigned char) (color.b);
        src.data[y * W * C + x * C + 1] = (unsigned char) (color.g);
        src.data[y * W * C + x * C + 2] = (unsigned char) (color.r);
    }

    static inline void setPixelHorWidth(const dcl::Mat &src, int x, int y, const dcl::Color &color, int thickness) {
        const int W = src.w();
        const int C = src.c();
        while (thickness--) {
            src.data[y * W * C + (x + thickness) * C + 0] = (unsigned char) (color.b);
            src.data[y * W * C + (x + thickness) * C + 1] = (unsigned char) (color.g);
            src.data[y * W * C + (x + thickness) * C + 2] = (unsigned char) (color.r);
        }
    }

    static inline void setPixeVerWidth(const dcl::Mat &src, int x, int y, const dcl::Color &color, int thickness) {
        const int W = src.w();
        const int C = src.c();
        while (thickness--) {
            src.data[(y + thickness) * W * C + x * C + 0] = (unsigned char) (color.b);
            src.data[(y + thickness) * W * C + x * C + 1] = (unsigned char) (color.g);
            src.data[(y + thickness) * W * C + x * C + 2] = (unsigned char) (color.r);
        }
    }

    static inline void setPixelAlpha(const dcl::Mat &src, int x, int y, const dcl::Color &color, float alpha) {
        const int W = src.w();
        const int C = src.c();
        src.data[y * W * C + x * C + 0] = (unsigned char) (src.data[y * W * C + x * C + 0] * (1 - alpha) + color.b * alpha);
        src.data[y * W * C + x * C + 1] = (unsigned char) (src.data[y * W * C + x * C + 1] * (1 - alpha) + color.g * alpha);
        src.data[y * W * C + x * C + 2] = (unsigned char) (src.data[y * W * C + x * C + 2] * (1 - alpha) + color.r * alpha);
    }

    static void lineBres(const dcl::Mat &src, const dcl::Point &p1, const dcl::Point &p2,
                         const dcl::Color &color, int thickness) {
        int x0 = p1.x;
        int y0 = p1.y;
        int x1 = p2.x;
        int y1 = p2.y;

        int p, twoDy, twoDyMinusDx, s1, s2;
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);

        if (dy > dx) {    //斜率大于1

            p = 2 * dx - dy;
            twoDy = 2 * dx;
            twoDyMinusDx = 2 * (dx - dy);

            if (y0 > y1) {  //斜率为负时 反转斜率
                std::swap(x0, x1);
                std::swap(y0, y1);
            }
            s1 = x1 > x0 ? 1 : -1;

            setPixelHorWidth(src, x0, y0, color, thickness);

            while (y0 < y1) {
                y0++;
                if (p < 0) {
                    p += twoDy;
                } else {
                    x0 += s1;
                    p += twoDyMinusDx;
                }
                setPixelHorWidth(src, x0, y0, color, thickness);
            }
        } else {
            p = 2 * dy - dx;
            twoDy = 2 * dy;
            twoDyMinusDx = 2 * (dy - dx);

            if (x0 > x1) { //斜率为负时 反转斜率
                std::swap(x0, x1);
                std::swap(y0, y1);
            }
            s2 = y1 > y0 ? 1 : -1;

            setPixeVerWidth(src, x0, y0, color, thickness);

            while (x0 < x1) {
                x0++;
                if (p < 0) {
                    p += twoDy;
                } else {
                    y0 += s2;
                    p += twoDyMinusDx;
                }
                setPixeVerWidth(src, x0, y0, color, thickness);
            }
        }
    }

    static void lineAnti_Wu(const dcl::Mat &src, const dcl::Point &p1, const dcl::Point &p2,
                            const dcl::Color &color) {
        int x0 = p1.x;
        int y0 = p1.y;
        int x1 = p2.x;
        int y1 = p2.y;

        int steep = abs(y1 - y0) > abs(x1 - x0);

        // swap the co-ordinates if slope > 1 or we
        // draw backwards
        if (steep) {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }

        //compute the slope
        float dx = x1 - x0;
        float dy = y1 - y0;
        float gradient = dy / dx;
        if (dx == 0.0)
            gradient = 1;

        int xpxl1 = x0;
        int xpxl2 = x1;
        float intersectY = y0;

        // main loop
        if (steep) {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                setPixelAlpha(src, int(intersectY), x, color, rfpart(intersectY));
                setPixelAlpha(src, int(intersectY) + 1, x, color, rfpart(intersectY));
                intersectY += gradient;
            }
        } else {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                setPixelAlpha(src, x, int(intersectY), color, rfpart(intersectY));
                setPixelAlpha(src, x, int(intersectY) + 1, color, rfpart(intersectY));
                intersectY += gradient;
            }
        }
    }

    // 吴小琳抗锯齿任意宽度
    static void lineAnti_WuMulti(const dcl::Mat &src, const dcl::Point &p1, const dcl::Point &p2,
                                 const dcl::Color &color, int thickness) {
        int x0 = p1.x;
        int y0 = p1.y;
        int x1 = p2.x;
        int y1 = p2.y;

        int steep = abs(y1 - y0) > abs(x1 - x0);

        // swap the co-ordinates if slope > 1, or we
        // draw backwards
        if (steep) {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }

        //compute the slope
        float dx = x1 - x0;
        float dy = y1 - y0;
        float gradient = dy / dx;
        if (dx == 0.0f)
            gradient = 1;

        int xpxl1 = x0;
        int xpxl2 = x1;
        float intersectY = y0;

        // main loop
        if (steep) {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                // pixel coverage is determined by fractional
                // part of y co-ordinate
                setPixelAlpha(src, int(intersectY), x, color, roundf(intersectY));
                for (int i = 1; i < thickness; ++i)
                    setPixel(src, int(intersectY) + i, x, color);
                setPixelAlpha(src, int(intersectY) + thickness, x, color, rfpart(intersectY));
                intersectY += gradient;
            }
        } else {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                // pixel coverage is determined by fractional
                // part of y co-ordinate
                setPixelAlpha(src, x, int(intersectY), color, roundf(intersectY));
                for (int i = 1; i < thickness; ++i)
                    setPixel(src, x, int(intersectY) + i, color);
                setPixelAlpha(src, x, int(intersectY) + thickness, color, rfpart(intersectY));
                intersectY += gradient;
            }
        }
    }

    // 高斯核进行权重划分  边缘锐化效果
    static void lineAnti_AreaWeight(const dcl::Mat &src, const dcl::Point &p1, const dcl::Point &p2,
                                    const dcl::Color &color, int thickness) {

#if 1
        //int weight[5][5] = { { 1, 2, 4, 2, 1 }, { 2, 5,6,5, 2 }, { 4, 6, 8,6,4 },{2,5,6,5,2 },{1,2,4,2,1} };
        //使用高斯累计核，优化速度
        int weight_sum[5][6] = {{0, 1, 3,  7,  9,  10},
                                {0, 2, 7,  13, 18, 20},
                                {0, 4, 10, 18, 24, 28},
                                {0, 2, 7,  13, 18, 20},
                                {0, 1, 3,  7,  9,  10}};
#else
        //使用平均核 查看效果  边缘平滑
        //int weight[5][5] = {{3, 3, 5, 3, 3},
        //                    {3, 3, 5, 3, 3},
        //                    {4, 4, 4, 4, 4},
        //                    {3, 3, 5, 3, 3},
        //                    {3, 3, 5, 3, 3}};
        int weight_sum[5][6] = {{0, 3, 6, 11, 14, 17},
                                {0, 3, 6, 11, 14, 17},
                                {0, 4, 8, 12, 16, 20},
                                {0, 3, 6, 11, 14, 17},
                                {0, 3, 6, 11, 14, 17}};
#endif

        int x0 = p1.x;
        int y0 = p1.y;
        int x1 = p2.x;
        int y1 = p2.y;

        int weight_1, weight_2, weight_3;
        float weight_temp;

        int steep = abs(y1 - y0) > abs(x1 - x0);

        // swap the co-ordinates if slope > 1 or we
        // draw backwards
        if (steep) {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }
        //compute the slope
        float dx = x1 - x0;
        float dy = y1 - y0;
        float gradient = dy / dx;
        if (dx == 0.0f)
            gradient = 1;

        int xpxl1 = x0;
        int xpxl2 = x1;
        float intersectY = y0;

        float temp;

        //上边界方程 y-y0 = k*(x -x0)
        //下边界方程 y-y0 = k*(x -x0)-r

        //因0<K<1 每条边界线只存在两种情况需要透明度计算
        //1.穿过一个像素
        //2.穿过两个像素
        //由此可知R+2为最大像素宽度，不需要抗锯齿的像素位置为 (Y-R +1) < int(Y) < (Y + 1 -2)
        //则边界线之间的距离小于3，则还会存在同时穿过一个像素

        // main loop
        if (steep) {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                weight_1 = 0;
                weight_2 = 0;
                weight_3 = 0;

                weight_temp = intersectY;
                for (int j = 0; j < 5; ++j) {
                    //上边界经过的像素点
                    weight_temp += gradient / 5;
                    temp = fabs(weight_temp - int(intersectY)) * 5;
                    if (temp > 5.0f) {
                        weight_2 += weight_sum[j][int(temp - 5.0f)];
                        weight_1 += weight_sum[j][5];    //当穿越时，下面的像素应该是满强度的

                        // 下边界经过的像素点 由直线的对称性得出
                        weight_3 += weight_sum[j][0];
                    } else {
                        weight_1 += weight_sum[j][int(temp)];

                        //下边界经过的像素点 由直线的对称性得出
                        weight_3 += weight_sum[j][int(5.0f - temp)];
                    }
                }
                // pixel coverage is determined by fractional
                // part of y co-ordinate
                if (temp > 5.0f) {
                    setPixelAlpha(src, int(intersectY) + 1, x, color, (88 - weight_2) / 88.0f);
                    setPixelAlpha(src, int(intersectY), x, color, weight_3 / 88.0f);
                    setPixelAlpha(src, int(intersectY) + thickness, x, color, weight_1 / 88.0f);
                    setPixelAlpha(src, int(intersectY) + thickness + 1, x, color, weight_2 / 88.0f);
                    for (int i = 1; i < thickness; ++i)
                        setPixel(src, int(intersectY) + i, x, color);
                } else {
                    setPixelAlpha(src, int(intersectY) + thickness, x, color, weight_1 / 88.0f);
                    setPixelAlpha(src, (int(intersectY)), x, color, weight_3 / 88.0f);
                    for (int i = 1; i < thickness; ++i)
                        setPixel(src, int(intersectY) + i, x, color);
                }
                intersectY += gradient;
            }
        } else {
            for (int x = xpxl1; x <= xpxl2; ++x) {
                weight_1 = 0;
                weight_2 = 0;
                weight_3 = 0;
                weight_temp = intersectY;
                for (int j = 0; j < 5; ++j) {
                    //上边界经过的像素点
                    weight_temp += gradient / 5;
                    temp = fabs((weight_temp - int(intersectY)) * 5);
                    if (temp > 5.0f) {
                        weight_2 += weight_sum[j][int(temp - 5.0f)];
                        weight_1 += weight_sum[j][5];    //当穿越时，下面的像素应该是满强度的

                        //下边界经过的像素点 由直线的对称性得出
                        weight_3 += weight_sum[j][0];
                    } else {
                        weight_1 += weight_sum[j][int(temp)];

                        //下边界经过的像素点 由直线的对称性得出
                        weight_3 += weight_sum[j][int(5.0f - temp)];
                    }
                }

                // pixel coverage is determined by fractional
                // part of y co-ordinate
                if (temp > 5.0f) {
                    setPixelAlpha(src, x, int(intersectY), color, weight_3 / 88.0f);
                    setPixelAlpha(src, x, int(intersectY) + 1, color, (88 - weight_2) / 88.0f);
                    setPixelAlpha(src, x, int(intersectY) + thickness, color, weight_1 / 88.0f);
                    setPixelAlpha(src, x, int(intersectY) + thickness + 1, color, weight_2 / 88.0f);
                    for (int i = 2; i < thickness; ++i)
                        setPixel(src, x, int(intersectY) + i, color);
                } else {
                    setPixelAlpha(src, x, int(intersectY), color, weight_3 / 88.0f);
                    setPixelAlpha(src, x, (int(intersectY)) + thickness, color, weight_1 / 88.0f);

                    for (int i = 1; i < thickness; ++i)
                        setPixel(src, x, int(intersectY) + i, color);
                }
                intersectY += gradient;
            }
        }
    }
}


#endif //DCL_WRAPPER_LINE_H
