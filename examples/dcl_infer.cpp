//
// Created by intellif on 23-5-23.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "models/base/net_operator.h"
#include "utils/device.h"
#include "utils/utils.h"
#include "base_type.h"


int main(int argc, char** argv) {
    if (argc != 4) {
        DCL_APP_LOG(DCL_ERROR, "input param num(%d) must be == 4,\n"
                               "\t1 - sdk.config, 2 - model file path, 3 - num_iter for inference", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *binFile = argv[2];
    const int num_iter = std::stoi(argv[3]);

    // sdk init
    dcl::deviceInit(sdkCfg);

    dcl::NetOperator net;
    std::vector<dcl::Tensor> vOutputTensors;

    // load model
    if (0 != net.load(binFile, false)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to load model");
        goto exit;
    }

    // inference
    for (int i=0; i<num_iter; ++i) {
        if (0 != net.inference(vOutputTensors)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to inference");
            goto exit;
        }
    }

    // unload
    if (0 != net.unload()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to unload model");
        goto exit;
    }

    exit:
    // sdk release
    dcl::deviceFinalize();
    return 0;
}