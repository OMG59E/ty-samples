//
// Created  on 22-8-24.
//

#ifndef DCL_WRAPPER_DEVICE_H
#define DCL_WRAPPER_DEVICE_H

#include <string>
#include "dcl.h"
#include "dcl_memory.h"

namespace ty {
    static int deviceInit(const std::string& config_file) {
        dclError ret = dclInit(config_file.c_str());
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dcl init failed, errorCode = %d", static_cast<int32_t>(ret));
            return -1;
        }
        DCL_APP_LOG(DCL_INFO, "dcl init success");

        int32_t majorVersion = 0, minorVersion = 0, patchVersion = 0;
        ret = dclrtGetVersion(&majorVersion, &minorVersion, &patchVersion);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "dcl get version failed, errorCode = %d", static_cast<int32_t>(ret));
            return -4;
        }
        DCL_APP_LOG(DCL_INFO, "tyhcp version: v%d.%d.%d", majorVersion, minorVersion, patchVersion);

        return 0;
    }

    static int deviceFinalize() {
        dclError ret = dclFinalize();
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "finalize dcl failed, errorCode = %d", static_cast<int32_t>(ret));
            return -1;
        }
        DCL_APP_LOG(DCL_INFO, "dcl finalize");

        return 0;
    }
}

#endif //DCL_WRAPPER_DEVICE_H
