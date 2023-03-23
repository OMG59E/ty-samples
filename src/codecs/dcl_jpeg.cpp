//
// Created  on 22-8-25.
//
#include <string>
#include <vector>
#include <fstream>

#include "dcl_jpeg.h"

namespace dcl {
    JpegDecoder::JpegDecoder(uint32_t chId)
        : chId_{chId} {
        mpiChannelDesc_.enType = DCL_PT_JPEG;
        mpiChannelDesc_.enMode = DCL_VIDEO_MODE_FRAME;
        mpiChannelDesc_.u32PicWidth = MAX_JPEG_IMAGE_W;
        mpiChannelDesc_.u32PicHeight = MAX_JPEG_IMAGE_H;
        mpiChannelDesc_.u32BufSize = MAX_JPEG_IMAGE_W * MAX_JPEG_IMAGE_H;
        mpiChannelDesc_.vbCnt = 4;
        mpiChannelDesc_.stVdecJpegAttr.enJpegFormat = DCL_JPG_COLOR_FMT_YCBCR420;

        dclError ret = dclmpiVdecCreateChn(chId_, &mpiChannelDesc_);
        if (ret != DCL_SUCCESS) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecCreateChn failed, ret = %d", ret);
            return;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecCreateChn done, chId=%d", chId_);
    }

    dcl::Mat JpegDecoder::decode(const char *filename) {
        dcl::Mat m;
        std::ifstream input(filename, std::ios::in | std::ios::binary | std::ios::ate);
        if (!input.is_open()) {
            DCL_APP_LOG(DCL_ERROR, "Cannot open image -> %s", filename);
            return m;
        }
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::vector<char> img_data;
        img_data.resize(static_cast<size_t>(file_size));
        input.read(img_data.data(), file_size);
        return decode(reinterpret_cast<unsigned char*>(img_data.data()), img_data.size());
    }

    dcl::Mat JpegDecoder::decode(unsigned char *data, size_t size) {
        dcl::Mat m;
        int height{0}, width{0};
        if (!getJpegImageSize(data, size, &width, &height)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to getJpegImageSize");
            return m;
        }

        decodeStreamDesc_.pu8Addr = data;
        decodeStreamDesc_.u32Len = size;
        decodeStreamDesc_.u64PTS = 40;
        decodeStreamDesc_.bEndOfFrame = 1;
        decodeStreamDesc_.bEndOfStream = 1;
        dclError ret = dclmpiVdecStartRecvStream(chId_);
        if (ret != DCL_SUCCESS) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecStartRecvStream failed, ret = %d", ret);
            return m;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecStartRecvStream done, chId=%d", chId_);

        ret = dclmpiVdecSendStream(chId_, &decodeStreamDesc_, nullptr, -1);
        if (ret != DCL_SUCCESS) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecSendStream failed, ret = %d", ret);
            dclmpiVdecStopRecvStream(chId_);
            return m;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecSendStream done, chId=%d", chId_);

        ret = dclmpiVdecGetFrame(chId_, &decodeFrameDesc_, nullptr, nullptr, -1);
        if (ret != DCL_SUCCESS) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecGetFrame failed, ret = %d", ret);
            dclmpiVdecStopRecvStream(chId_);
            return m;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecGetFrame done, chId=%d", chId_);

        void *mpiOutputBuffer = decodeFrameDesc_.vFrame.virtAddr[0];
        auto mpiOutputSize = decodeFrameDesc_.vFrame.widthStride[0] * decodeFrameDesc_.vFrame.heightStride[0] * 3 / 2;

        // TODO enJpegFormat <==> dclPixelFormat
        pixelFormat_t pixelFormat{DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_420};
        switch (mpiChannelDesc_.stVdecJpegAttr.enJpegFormat) {
            case DCL_JPG_COLOR_FMT_YCBCR400:
                pixelFormat = DCL_PIXEL_FORMAT_YUV_400;
                break;
            case DCL_JPG_COLOR_FMT_YCBCR420:
                break;
            default:
                DCL_APP_LOG(DCL_ERROR, "Failed to convert enJpegFormat to dclPixelFormat, ret = %d", ret);
                return m;
        }

        int align_height = ((height + 15) / 16) * 16;
        int align_width = ((width + 15) / 16) * 16;
        m.create(align_height, align_width, pixelFormat);
        // update
        m.original_height = height;
        m.original_width = width;

        ret = dclrtMemcpy(m.data, m.size(), mpiOutputBuffer, mpiOutputSize, DCL_MEMCPY_DEVICE_TO_DEVICE);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dclrtMemcpy failed, ret = %d", ret);
            m.free();
            return m;
        }

        ret = dclmpiVdecReleaseFrame(chId_, &decodeFrameDesc_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecReleaseFrame failed, ret = %d", ret);
            m.free();
            return m;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecReleaseFrame done, chId=%d", chId_);

        ret = dclmpiVdecStopRecvStream(chId_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecStopRecvStream failed, ret = %d", ret);
            return m;
        }
        DCL_APP_LOG(DCL_INFO, "dclmpiVdecStopRecvStream done, chId=%d", chId_);

        return m;
    }

    void JpegDecoder::release() const {
        dclError ret = dclmpiVdecDestroyChn(chId_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dclmpiVdecDestroyChn failed, ret = %d", ret);
            return;
        }
    }

    bool JpegDecoder::getJpegImageSize(unsigned char *data, size_t size, int *width, int *height) {
        bool bRet = false;

        unsigned char *p = data;

        if (p[0] != 0xFF || p[1] != 0xD8)
            return false;

        p += 2;

        do {
            if (p[0] == 0xFF && p[1] == 0xD9)
                break;

            if (p - data >= size)
                break;

            if (p[0] == 0xFF) {
                if (p[1] == 0xC0) {
                    p += 5;
                    unsigned char bySize[2] = {0};
                    bySize[0] = p[1];
                    bySize[1] = p[0];

                    unsigned short imgHeight;
                    memcpy(&imgHeight, bySize, 2);

                    bySize[0] = p[3];
                    bySize[1] = p[2];

                    unsigned short imgWidth;
                    memcpy(&imgWidth, bySize, 2);

                    *width = imgWidth;
                    *height = imgHeight;

                    bRet = true;

                    break;
                } else if ((p[1] >= 0xE0 && p[1] <= 0xFF)
                           || p[1] == 0xDB
                           || p[1] == 0xC4
                           || p[1] == 0xDA) {
                    unsigned char byLen[2] = {0};
                    byLen[0] = p[3];
                    byLen[1] = p[2];
                    unsigned short len1;
                    memcpy(&len1, byLen, 2);
                    p += 2;
                    p += len1;
                } else {
                    p++;
                }
            } else {
                p++;
            }
        } while (true);

        return bRet;
    }
}