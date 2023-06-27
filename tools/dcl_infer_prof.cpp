//
// Created on 23-5-11.
//
#include <thread>
#include <chrono>
#include "base_type.h"
#include "models/base/net_operator.h"
#include "utils/concurrentqueue.h"
#include "utils/resize.h"
#include "utils/device.h"
#include "utils/macro.h"

using namespace moodycamel;

typedef struct {
    uint64_t preprocess_time{0};  // us
    uint64_t inference_time{0};  // us
    uint64_t total_time() const { return preprocess_time + inference_time; }
} counter_t;

void infer(dcl::NetOperator *net, ConcurrentQueue<dcl::Mat> &queue, counter_t& counter,
           bool isClone) {
    dcl::NetOperator *p = isClone ? net->clone() : net;
    dcl::Mat img;
    std::vector<dcl::Tensor> vOutputTensors;
    std::vector<dcl::input_t> &inputs = p->getInputs();
    while (true) {
        if (!queue.try_dequeue(img))
            break;
        // preprocess
        auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        dcl::Mat m;
        m.data = static_cast<unsigned char *>(inputs[0].data);
        m.phyAddr = inputs[0].phyAddr;
        m.channels = inputs[0].c();
        m.height = inputs[0].h();
        m.width = inputs[0].w();
        m.pixelFormat = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
        dclResizeCvtPaddingOp(img, m, dcl::NONE, 114);
        auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        counter.preprocess_time += t1 - t0;
        // inference
        vOutputTensors.clear();
        p->inference(vOutputTensors);
        auto t2 = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        counter.inference_time += t2 - t1;
    }
    if (isClone) {
        p->unload();
        SAFE_FREE(p);
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("input param num(%d) must be == 6,\n"
               "\t1 - sdk.config, 2 - input image path, 3 - model file path, 4 - num sample, 5 - num thread", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *modelFile = argv[3];
    const int numSamples = std::stoi(argv[4]);
    const int numThreads = std::stoi(argv[5]);

    if (numSamples <= 0) {
        DCL_APP_LOG(DCL_ERROR, "numSamples must be > 0");
        return -1;
    }

    if (numThreads <= 0) {
        DCL_APP_LOG(DCL_ERROR, "numThreads must be > 0");
        return -1;
    }

    // sdk init
    dcl::deviceInit(sdkCfg);

    std::vector<counter_t> spans;
    uint64_t t_start{0}, t_end{0};
    std::vector<std::thread> vt;
    ConcurrentQueue<dcl::Mat> queue(numSamples);
    dcl::Mat img;
    dcl::NetOperator net;
    uint64_t max_span = 0;
    int32_t idx = -1;

    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to read img, maybe filepath not exist -> %s", imgPath);
        goto exit;
    }

    net.load(modelFile);
    if (net.getInputs().size() != 1) {
        DCL_APP_LOG(DCL_ERROR, "Not support multi-input");
        goto exit;
    }

    img.create(src.rows, src.cols, DCL_PIXEL_FORMAT_BGR_888_PACKED);
    memcpy(img.data, src.data, img.size());

    for (int n = 0; n < numSamples; ++n) {
        dcl::Mat m;
        m.channels = img.c();
        m.width = img.w();
        m.height = img.h();
        m.data = img.data;
        m.phyAddr = img.phyAddr;
        m.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PACKED;
        m.own = false;
        if (!queue.try_enqueue(m)) {
            DCL_APP_LOG(DCL_ERROR, "Failed to enqueue");
            goto exit;
        }
    }

    vt.clear();
    spans.resize(numThreads);
    for (int n = 0; n < numThreads; ++n) {
        bool use = n == 0;
        vt.emplace_back(infer, &net, std::ref(queue), std::ref(spans[n]), !use);
    }

    for (auto &t: vt)
        t.join();

    // find max thread time
    for (int n=0; n<numThreads; ++n) {
        if (spans[n].total_time() > max_span) {
            max_span = spans[n].total_time();
            idx = n;
        }
    }
    DCL_APP_LOG(DCL_INFO, "samples: %d, preprocess: %.3fms, infer: %.3fms, preprocess + infer: %.3fms", numSamples,
                spans[idx].preprocess_time / 1000.0f, spans[idx].inference_time / 1000.0f, (max_span) / 1000.0f);
    DCL_APP_LOG(DCL_INFO, "average infer: %.3fms, fps: %.3f",
                spans[idx].inference_time / 1000.0f / numSamples, numSamples * 1e6 / float(max_span));
    net.unload();

    exit:
    src.release();
    img.free();
    dcl::deviceFinalize();
    return 0;
}