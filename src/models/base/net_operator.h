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

namespace dcl {
    typedef struct {
        int idx{-1};
        int aippIdx{-1};
        dclmdlIODims dim{{0}};
        void *aippData{nullptr};
        size_t aippSize{0};
    } input_t;

    class NetOperator {
    public:
        NetOperator() = default;
        ~NetOperator() = default;

        /**
         * load model from file
         * @param modelPath
         * @param enableAipp
         * @return
         */
        int load(const std::string &modelPath, bool enableAipp = true);

        /**
         * model inference for signal input
         * @param image
         * @param vOutputTensors
         * @param handle
         * @return
         */
        int inference(const dcl::Mat &image,
                      std::vector<dcl::Tensor> &vOutputTensors);

        /**
         * model inference for multi-input
         * @param images
         * @param vOutputTensors
         * @param handle
         * @return
         */
        int inference(const std::vector<dcl::Mat> &images,
                      std::vector<dcl::Tensor> &vOutputTensors);

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
        int getInputNum() const;

        /**
         * get output num
         * @return
         */
        int getOutputNum() const;

    private:
        int createModelDesc();

        int destroyModelDesc();

        int createInputDataset();

        int updateDataBuffer(const std::vector<dcl::Mat> &images);

        int createOutputDataset();

        int destroyOutputDataset();

        int destroyInputDataset();

        int findAippDataIdx(int aippIdx) const;

    public:
        std::mutex rw_mutex_;

    private:
        dclmdlDataset *inputDataset_{nullptr};
        dclmdlDataset *outputDataset_{nullptr};
        size_t nbNumInput_{0};
        size_t nbNumOutput_{0};
        size_t nbAippInput_{0};

        std::vector<input_t> vInputs_;
        std::vector<dclmdlIODims> vInputDims_;
        std::vector<dclmdlIODims> vOutputDims_;

        dclprofConfig *profCfgP_{nullptr};
        bool enableProf_{false};
        bool enableAipp_{true};
        bool loadFlag_{false};  // loadModel
        size_t workSize_{0};
        size_t weightSize_{0};
        void *workPtr_{nullptr}; // model work memory buffer
        void *weightPtr_{nullptr}; // model weight memory buffer
        uint32_t id_{0};
        dclmdlDesc *desc_{nullptr};
    };
}

#endif //DCL_WRAPPER_NET_OPERATOR_H
