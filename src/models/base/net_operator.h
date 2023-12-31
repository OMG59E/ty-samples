//
// Created  on 22-8-24.
//

#ifndef DCL_WRAPPER_NET_OPERATOR_H
#define DCL_WRAPPER_NET_OPERATOR_H

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <chrono>

#include "dcl.h"
#include "dcl_prof.h"
#include "base_type.h"

using namespace std::chrono;

#define MAX_IMAGE_C 3
#define MAX_IMAGE_H 1080
#define MAX_IMAGE_W 1920
#define MAX_IMAGE_SIZE MAX_IMAGE_C*MAX_IMAGE_H*MAX_IMAGE_W

namespace ty {
    typedef struct {
        int idx{-1};
        void *data{nullptr};
        uint64_t phyAddr{0};
        dclmdlIODims dim{{0}};
        size_t dataSize{0};
        int height{0};
        int width{0};
        int original_height{0};
        int original_width{0};
        int channels{0};
        pixelFormat_t pixelFormat{DCL_PIXEL_FORMAT_BGR_888_PACKED};

        int aippIdx{-1};
        void *aippData{nullptr};
        size_t aippSize{0};

        bool hasAipp() const { return aippIdx != -1; }

        void update(int c, int h, int w, pixelFormat_t _pixelFormat) {
            dataSize = c * h * w;
            height = h;
            width = w;
            channels = c;
            original_width = width;
            original_height = height;
            pixelFormat = _pixelFormat;
        }

        int c() const { return channels; }
        int h() const { return height; }
        int w() const { return width; }
    } input_t;

    class NetOperator {
    public:
        NetOperator() = default;
        ~NetOperator() = default;

        NetOperator(uint32_t modelId);

        /**
         * load model from file
         * @param modelPath
         * @param enableAipp
         * @return
         */
        int load(const std::string &modelPath, bool enableAipp = true);

        /**
         * model inference for multi-input
         * @param images
         * @param vOutputTensors
         * @param handle
         * @return
         */
        int inference(std::vector<ty::Tensor> &vOutputTensors);

        /**
         * unload model
         * @return
         */
        int unload();

        /**
         * print infer info
         * @return
         */
        int setProfile(const std::string& profilePath);

        /**
         * get input num
         * @return
         */
        int getInputNum() const { return nbNumInput_; };

        /**
         * get output num
         * @return
         */
        int getOutputNum() const { return nbNumOutput_; };


        std::vector<input_t>& getInputs() { return vInputs_; };

        NetOperator* clone() const;

    private:
        int initInputOutput();

        int createModelDesc();

        int destroyModelDesc();

        int createInputDataset();

        int createOutputDataset();

        int destroyOutputDataset();

        int destroyInputDataset();

    public:
        static uint32_t ref_count_;
        static std::mutex rw_mutex_;

    private:
        dclmdlDataset *inputDataset_{nullptr};
        dclmdlDataset *outputDataset_{nullptr};
        int nbNumInput_{0};
        int nbNumOutput_{0};
        int nbAippInput_{0};

        std::vector<input_t> vInputs_;
        std::vector<dclmdlIODims> vInputDims_;
        std::vector<dclmdlIODims> vOutputDims_;

        dclprofConfig *profCfgP_{nullptr};
        bool enableProf_{false};
        // bool enableAipp_{true};
        uint32_t modelId_{0};
        dclmdlDesc *desc_{nullptr};
    };
}

#endif //DCL_WRAPPER_NET_OPERATOR_H
