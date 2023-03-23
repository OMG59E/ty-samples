//
// Created  on 22-8-25.
//

#ifndef DCL_WRAPPER_DCL_JPEG_H
#define DCL_WRAPPER_DCL_JPEG_H

#include "dcl_mpi.h"
#include "base_type.h"

#define MAX_JPEG_IMAGE_H 4096
#define MAX_JPEG_IMAGE_W 4096

namespace dcl {
    class JpegDecoder {
    public:
        explicit JpegDecoder(uint32_t chId = 0);
        ~JpegDecoder() = default;

        void release() const;

        dcl::Mat decode(const char *filename);
        dcl::Mat decode(unsigned char* data, size_t size);

    private:
        static bool getJpegImageSize(unsigned char *data, size_t size, int *width, int *height);

    private:
        uint32_t chId_{0};
        dclmpiVdecChnAttr mpiChannelDesc_{};
        dclVideoFrameInfo decodeFrameDesc_{};
        dclmpiVdecStream decodeStreamDesc_{};
    };
}

#endif //DCL_WRAPPER_DCL_JPEG_H
