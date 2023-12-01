//
// Created  on 22-8-24.
//

#ifndef DCL_WRAPPER_UTILS_H
#define DCL_WRAPPER_UTILS_H
#include <sys/stat.h>
#include <fstream>
#include <string>
// dcl
#include "dcl.h"
#include "dcl_base.h"


static int checkPathIsFile(const std::string &fileName) {
#if defined(_MSC_VER)
    DWORD bRet = GetFileAttributes((LPCSTR)fileName.c_str());
    if (bRet == FILE_ATTRIBUTE_DIRECTORY) {
        DCL_APP_LOG(DCL_ERROR, "%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
#else
    struct stat sBuf{};
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        DCL_APP_LOG(DCL_ERROR, "failed to get file");
        return -1;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        DCL_APP_LOG(DCL_ERROR, "%s is not a file, please enter a file", fileName.c_str());
        return -2;
    }
#endif
    return 0;
}


static int readBinFile(const std::string &fileName, void *&inputBuff, uint32_t &fileSize) {
    if (checkPathIsFile(fileName) != 0) {
        DCL_APP_LOG(DCL_ERROR, "%s is not a file", fileName.c_str());
        return -1;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (!binFile.is_open()) {
        DCL_APP_LOG(DCL_ERROR, "open file %s failed", fileName.c_str());
        return -2;
    }

    binFile.seekg(0, std::ifstream::end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        DCL_APP_LOG(DCL_ERROR, "binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return -3;
    }
    binFile.seekg(0, std::ifstream::beg);

    dclError ret = dclrtMalloc(&inputBuff, binFileBufferLen, DCL_MEM_MALLOC_NORMAL_ONLY);
    if (DCL_SUCCESS != ret) {
        DCL_APP_LOG(DCL_ERROR, "malloc device buffer failed. size is %u, errorCode is %d",
                    binFileBufferLen, static_cast<int32_t>(ret));
        binFile.close();
        return -4;
    }
    binFile.read(static_cast<char *>(inputBuff), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return 0;
}

static ty::Mat cvMatToDclMat(const cv::Mat& cvMat) {
    ty::Mat dclMat;
    dclMat.data = cvMat.data;
    dclMat.channels = cvMat.channels();
    dclMat.height = cvMat.rows;
    dclMat.width = cvMat.cols;
    dclMat.original_height = cvMat.rows;
    dclMat.original_width = cvMat.cols;
    dclMat.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PACKED;
    dclMat.own = false;
    return dclMat;
}

#endif //DCL_WRAPPER_UTILS_H
