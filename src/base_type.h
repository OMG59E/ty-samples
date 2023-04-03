//
// Created  on 22-8-24.
//

#ifndef DCL_WRAPPER_BASE_TYPE_H
#define DCL_WRAPPER_BASE_TYPE_H

#include "dcl.h"
#include "utils/macro.h"
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>

namespace dcl {
    typedef dclPixelFormat pixelFormat_t;

    struct Mat {
        unsigned char *data{nullptr};
        uint64_t phyAddr{0};
        int channels{0};
        int height{0};  // 16 pixel align
        int width{0};   // 16 pixel align
        bool own{false};
        pixelFormat_t pixelFormat{DCL_PIXEL_FORMAT_BGR_888};

        int original_height{0};
        int original_width{0};

        Mat() = default;
        // ~Mat() { free(); }

        Mat(int _height, int _width, pixelFormat_t _pixelFormat) {
            create(_height, _width, _pixelFormat);
        }

        int create(int _height, int _width, pixelFormat_t _pixelFormat) {
            if (data) {
                DCL_APP_LOG(DCL_ERROR, "data != nullptr");
                return -1;
            }
            height = _height;
            width = _width;
            original_height = _height;
            original_width = _width;
            pixelFormat = _pixelFormat;
            switch (pixelFormat) {
                case DCL_PIXEL_FORMAT_BGR_888:
                case DCL_PIXEL_FORMAT_RGB_888:
                case DCL_PIXEL_FORMAT_BGR_888_PLANAR:
                case DCL_PIXEL_FORMAT_RGB_888_PLANAR:
                case DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
                case DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
                    channels = 3;
                    break;
                case DCL_PIXEL_FORMAT_YUV_400:
                case DCL_PIXEL_FORMAT_F32C1:
                    channels = 1;
                    break;
                default:
                    DCL_APP_LOG(DCL_ERROR, "Not support pixel_format: %d", pixelFormat);
                    return -2;
            }

            own = true;
            dclError ret = dclrtMallocEx((void**)&data, &phyAddr, size(), 16, DCL_MEM_MALLOC_NORMAL_ONLY);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "dclrtMallocEx failed, channels: %d height: %d width: %d",
                            channels, height, width);
                return -3;
            }
            return 0;
        }

        void free() {
            if (own)
                DCLRT_FREE(data);
        }

        bool empty() const { return !data; }

        unsigned char *ptr() const { return data; }

        int c() const { return channels; }

        int h() const { return height; }

        int w() const { return width; }

        size_t size() const {
            switch (pixelFormat) {
                case DCL_PIXEL_FORMAT_BGR_888:
                case DCL_PIXEL_FORMAT_RGB_888:
                case DCL_PIXEL_FORMAT_BGR_888_PLANAR:
                case DCL_PIXEL_FORMAT_RGB_888_PLANAR:
                case DCL_PIXEL_FORMAT_YUV_400:  // gray
                    return channels * height * width;
                case DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
                case DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
                    return height * width * channels / 2;
                case DCL_PIXEL_FORMAT_F32C1:
                    return height * width * channels * sizeof(float);
                default:
                    DCL_APP_LOG(DCL_ERROR, "Not support pixel_format: %d", pixelFormat);
                    return 0;
            }
        }
    };

    typedef dclFormat dataLayout_t;
    typedef dclDataType dclDataType_t;

    struct Tensor {
        float *data{nullptr};
        int nbDims{0};
        int d[8]{};
        dataLayout_t dataLayout{DCL_FORMAT_NCHW};
        dclDataType_t dataType{DCL_FLOAT};

        int n() const { return d[0]; }

        int c() const { return dataLayout == DCL_FORMAT_NCHW ? d[1] : d[3]; }

        int h() const { return dataLayout == DCL_FORMAT_NCHW ? d[2] : d[1]; }

        int w() const { return dataLayout == DCL_FORMAT_NCHW ? d[3] : d[2]; }

        size_t size() const {
            if (0 == nbDims)
                return 0;
            size_t length = 1;
            for (int i = 0; i < nbDims; ++i)
                length *= d[i];
            return length;
        }
    };

    struct Box {
        int x1{0};
        int y1{0};
        int x2{0};
        int y2{0};

        Box() = default;

        Box(int _x1, int _y1, int _x2, int _y2) {
            x1 = _x1;
            y1 = _y1;
            x2 = _x2;
            y2 = _y2;
        }

        int w() const { return x2 - x1 + 1; }

        int h() const { return y2 - y1 + 1; }

        int x() const { return x1; }

        int y() const { return y1; }

        int cx() const { return (x1 + x2) / 2; }

        int cy() const { return (y1 + y2) / 2; }
    };

    struct Size {
        int h{0}, w{0};
        Size() = default;
        Size(int _h, int _w) { h = _h; w = _w;}
    };
    
    struct Point {
        int x{0};
        int y{0};
        float score{0.0f};
        Point() = default;
        Point(int _x, int _y) {
            x = _x;
            y = _y;
        }
    };

    struct Color {
        uint8_t b{0};
        uint8_t g{0};
        uint8_t r{0};

        Color() = default;
        Color(uint8_t _b, uint8_t _g, uint8_t _r) {
            set(_b, _g, _r);
        }

        void set(uint8_t _b, uint8_t _g, uint8_t _r) {
            b = _b;
            g = _g;
            r = _r;
        }
    };

    typedef std::vector<cv::Point> contour_t;

    typedef struct {
        float conf{0.0f};
        int cls{-1};  // cls index
        std::string name; // cls_name
        Box box;
        Point pts[5]{};  // 5 landmark
        Point kpts[17]{}; // 17-keypoints
        float mask[32]{};
        dcl::Mat prob;
        std::vector<contour_t> contours;
    } detection_t;

    typedef struct {
        int cls{-1};  // cls index
        std::string name; // cls_name
        float conf{0.0f};
    } classification_t;
}
#endif //DCL_WRAPPER_BASE_TYPE_H
