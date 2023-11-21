//
// Created  on 22-8-24.
//
#include "net_operator.h"
#include "utils/utils.h"
#include "utils/macro.h"

namespace dcl {
    uint32_t NetOperator::ref_count_ = 0;
    std::mutex NetOperator::rw_mutex_;

    NetOperator::NetOperator(uint32_t modelId) {
        modelId_ = modelId;
        rw_mutex_.lock();
        ref_count_++;
        rw_mutex_.unlock();
    }

    int NetOperator::initInputOutput() {
        if (createModelDesc() != 0)
            return -5;

        if (createInputDataset() != 0) {
            DCL_APP_LOG(DCL_ERROR, "Failed to createInputDataset, modelId %d", modelId_);
            return -6;
        }

        if (createOutputDataset() != 0) {
            DCL_APP_LOG(DCL_ERROR, "Failed to createOutputDataset, modelId %d", modelId_);
            return -7;
        }
        return 0;
    }

    int NetOperator::load(const std::string &modelPath, bool enableAipp) {
        dclError ret = dclmdlLoadFromFile(modelPath.c_str(), &modelId_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "Failed to load model, model file is %s, errorCode is %d",
                        modelPath.c_str(), static_cast<int32_t>(ret));
            return -1;
        }
        rw_mutex_.lock();
        ref_count_++;
        rw_mutex_.unlock();
        DCL_APP_LOG(DCL_INFO, "load model %s success", modelPath.c_str());
        return initInputOutput();
    }

    int NetOperator::unload() {
        dclError ret;
        rw_mutex_.lock();
        if (1 == ref_count_) {
            rw_mutex_.unlock();
            ret = dclmdlUnload(modelId_);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "unload model failed, modelId is %u, errorCode is %d",
                            modelId_, static_cast<int32_t>(ret));
                return -1;
            }
        }
        rw_mutex_.unlock();
        rw_mutex_.lock();
        ref_count_--;
        rw_mutex_.unlock();

        destroyInputDataset();
        for (auto& input : vInputs_) {
            DCLRT_FREE(input.aippData);
            DCLRT_FREE(input.data);
        }

        destroyOutputDataset();
        destroyModelDesc();

        DCL_APP_LOG(DCL_INFO, "unload model success, modelId is %u", modelId_);
        modelId_ = 0;

        if (enableProf_) {
            ret = dclprofStop(profCfgP_);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to dclprofStop, ret: %d", ret);
                return -2;
            }
            ret = dclprofDestroyConfig(profCfgP_);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to dclprofDestroyConfig, ret: %d", ret);
                return -3;
            }
            ret = dclprofFinalize();
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to dclprofFinalize, ret: %d", ret);
                return -4;
            }
            enableProf_ = false;
        }
        return 0;
    }

    NetOperator* NetOperator::clone() const {
        auto* p = new NetOperator(modelId_);
        dclError ret = p->initInputOutput();
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "Failed to initInputOutput, ret: %d", ret);
            return nullptr;
        }
        return p;
    }

    int NetOperator::destroyOutputDataset() {
        if (outputDataset_) {
            for (size_t i = 0; i < dclmdlGetDatasetNumBuffers(outputDataset_); ++i) {
                dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(outputDataset_, i);
                void *data = dclGetDataBufferAddr(dataBuffer);
                DCLRT_FREE(data);
                DCLMDL_DATABUFFER_FREE(dataBuffer);
            }
            DCLMDL_DATASET_FREE(outputDataset_);
        }
        return 0;
    }

    int NetOperator::destroyInputDataset() {
        if (inputDataset_) {
            for (size_t i = 0; i < dclmdlGetDatasetNumBuffers(inputDataset_); ++i) {
                dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(inputDataset_, i);
                DCLMDL_DATABUFFER_FREE(dataBuffer);
            }
            DCLMDL_DATASET_FREE(inputDataset_);
        }
        return 0;
    }

    int NetOperator::createModelDesc() {
        desc_ = dclmdlCreateDesc();
        if (!desc_) {
            DCL_APP_LOG(DCL_ERROR, "create model description failed");
            return -1;
        }

        dclError ret = dclmdlGetDesc(desc_, modelId_);
        if (ret != DCL_ERROR_NONE) {
            DCL_APP_LOG(DCL_ERROR, "get model description failed, modelId is %u, errorCode is %d",
                        modelId_, static_cast<int32_t>(ret));
            return -2;
        }
        DCL_APP_LOG(DCL_INFO, "create model description success");
        return 0;
    }

    int NetOperator::destroyModelDesc() {
        DCLMDL_DESC_FREE(desc_);
        DCL_APP_LOG(DCL_INFO, "destroy model description success");
        return 0;
    }

    int NetOperator::inference(std::vector<dcl::Tensor> &vOutputTensors) {
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        dclError ret = dclmdlExecute(modelId_, inputDataset_, outputDataset_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "Failed to execute model, modelId is %u, errorCode is %d",
                        modelId_, static_cast<int32_t>(ret));
            return -3;
        }

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        vOutputTensors.clear();
        vOutputTensors.resize(nbNumOutput_);
        for (int i = 0; i < nbNumOutput_; ++i) {
            dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(outputDataset_, i);
            // void *data = dclGetDataBufferAddr(dataBuffer);
            vOutputTensors[i].data = dclGetDataBufferAddr(dataBuffer);
            vOutputTensors[i].nbDims = int(vOutputDims_[i].dimCount);
            for (int j = 0; j < vOutputTensors[i].nbDims; ++j)
                vOutputTensors[i].d[j] = int(vOutputDims_[i].dims[j]);
            dclFormat format = dclmdlGetOutputFormat(desc_, i);
            dclDataType datatype = dclmdlGetOutputDataType(desc_, i);
            vOutputTensors[i].dataType = datatype;
            if (DCL_FORMAT_NCHW == format || DCL_FORMAT_UNDEFINED == format)
                vOutputTensors[i].dataLayout = DCL_FORMAT_NCHW;
            else if (DCL_FORMAT_NHWC == format)
                vOutputTensors[i].dataLayout = DCL_FORMAT_NHWC;
            else {
                DCL_APP_LOG(DCL_ERROR, "Not support format %d", format);
                return -4;
            }
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<float, std::micro> tp0 = t1 - t0;
        duration<float, std::micro> tp1 = t2 - t1;
        DCL_APP_LOG(DCL_INFO, "Execute: %.3fms, GetOutput: %.3fms", tp0.count() / 1000.0f, tp1.count() / 1000.0f);
        return 0;
    }

    int NetOperator::createInputDataset() {
        //
        inputDataset_ = dclmdlCreateDataset();
        if (!inputDataset_) {
            DCL_APP_LOG(DCL_ERROR, "Failed to create input dataset");
            return -2;
        }

        dclError ret;
        vInputs_.clear();
        vInputDims_.clear();
        for (int n=0; n<dclmdlGetNumInputs(desc_); ++n) {
            nbNumInput_++;
            dclDataBuffer *inputData = dclCreateDataBuffer(nullptr, 0);
            ret = dclmdlAddDatasetBuffer(inputDataset_, inputData);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to add input dataset buffer, error code: %d", static_cast<int32_t>(ret));
                DCLMDL_DATABUFFER_FREE(inputData);
                return -3;
            }

            input_t input;
            input.idx = n;
            dclmdlIODims dims;
            memset(&dims, 0, sizeof(dims));
            ret = dclmdlGetInputDims(desc_, n, &dims);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to get input dims, error code: %d", static_cast<int32_t>(ret));
                return -4;
            }
            size_t size = 1;
            std::string shape;
            for (int d=0; d<dims.dimCount; ++d) {
                shape += std::to_string(dims.dims[d]);
                shape += " ";
                size *= dims.dims[d];
            }

            input.dataSize = size;
            if (dims.dimCount == 4) {
                input.channels = int(dims.dims[1]);
                input.height = int(dims.dims[2]);
                input.width = int(dims.dims[3]);
                input.original_height = input.height;
                input.original_width = input.width;
            } else {
                DCL_APP_LOG(DCL_ERROR, "Not support yet other dims: %d", dims.dimCount);
                return -8;
            }

            DCL_APP_LOG(DCL_INFO, "create data input[%d] success, nbDim: %d, shape: %s, dtype: %d, aipp: %d",
                        n, dims.dimCount, shape.c_str(), dclmdlGetInputDataType(desc_, n), input.aippIdx);

            ret = dclrtMallocEx(&(input.data), &(input.phyAddr), input.dataSize, 16, DCL_MEM_MALLOC_NORMAL_ONLY);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to malloc data buffer without aipp, error code: %d", ret);
                return -5;
            }
            input.dim = dims;
            vInputs_.emplace_back(input);
            vInputDims_.emplace_back(dims);
        }

        // bind aipp and data
        for (auto &input : vInputs_) {
            if (input.hasAipp()) {
                dclDataBuffer* aippBuffer = dclmdlGetDatasetBuffer(inputDataset_, input.aippIdx);
                ret = dclUpdateDataBuffer(aippBuffer, input.aippData, input.aippSize);
                if (DCL_SUCCESS != ret) {
                    DCL_APP_LOG(DCL_ERROR, "Failed to update aipp buffer, error code: %d", ret);
                    return -7;
                }
            }

            dclDataBuffer* dataBuffer = dclmdlGetDatasetBuffer(inputDataset_, input.idx);
            ret = dclUpdateDataBuffer(dataBuffer, input.data, input.dataSize);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to update data buffer, error code: %d", ret);
                return -8;
            }
        }
        return 0;
    }

    int NetOperator::createOutputDataset() {
        if (!desc_) {
            DCL_APP_LOG(DCL_ERROR, "no model description, create ouput failed");
            return -1;
        }

        dclError ret;
        // collect output infos
        nbNumOutput_ = int(dclmdlGetNumOutputs(desc_));
        for (int i = 0; i < nbNumOutput_; ++i) {
            dclmdlIODims dims;
            ret = dclmdlGetOutputDims(desc_, i, &dims);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to get input dims, errorCode is %d", static_cast<int32_t>(ret));
                return -1;
            }
            vOutputDims_.emplace_back(dims);
        }

        outputDataset_ = dclmdlCreateDataset();
        if (!outputDataset_) {
            DCL_APP_LOG(DCL_ERROR, "can't create dataset, create output failed");
            return -2;
        }

        for (int i = 0; i < nbNumOutput_; ++i) {
            size_t outputSize = dclmdlGetOutputSizeByIndex(desc_, i);
            void *outputBuffer = nullptr;
            dclError ret = dclrtMalloc(&outputBuffer, outputSize, DCL_MEM_MALLOC_NORMAL_ONLY);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "can't malloc buffer, create output failed, size is %zu, errorCode is %d",
                            outputSize, static_cast<int32_t>(ret));
                DCLMDL_DATASET_FREE(outputDataset_);
                return -3;
            }

            dclDataBuffer *outputData = dclCreateDataBuffer(outputBuffer, outputSize);
            if (!outputData) {
                DCL_APP_LOG(DCL_ERROR, "can't create data buffer, create output failed");
                DCLMDL_DATASET_FREE(outputDataset_);
                DCLRT_FREE(outputBuffer);
                return -4;
            }

            ret = dclmdlAddDatasetBuffer(outputDataset_, outputData);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "can't add data buffer, create output failed");
                DCLMDL_DATASET_FREE(outputDataset_);
                DCLRT_FREE(outputBuffer);
                DCLMDL_DATABUFFER_FREE(outputData);
                return -5;
            }
            DCL_APP_LOG(DCL_INFO, "add output[%zu] size: %zu", i, outputSize);
        }
        DCL_APP_LOG(DCL_INFO, "create output success");
        return 0;
    }

    int NetOperator::setProfile(const std::string &profilePath) {
        dclError ret = dclprofInit(profilePath.c_str(), profilePath.size());
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "profile init failed, errorCode = %d", static_cast<int32_t>(ret));
            return -1;
        }
        DCL_APP_LOG(DCL_INFO, "profile init success");

        profCfgP_ = dclprofCreateConfig(nullptr, 0, DCL_AICORE_NONE,
                                                     nullptr, DCL_PROF_DCL_API | DCL_PROF_TASK_TIME);
        if (!profCfgP_) {
            DCL_APP_LOG(DCL_ERROR, "Failed to create profile, ret: %d", ret);
            return -2;
        }
        DCL_APP_LOG(DCL_INFO, "profile create success");
        enableProf_ = true;
        if (enableProf_) {
            ret = dclprofStart(profCfgP_);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to start profile, ret: %d", ret);
                DCL_PROFILE_FREE(profCfgP_);
                return -2;
            }
            DCL_APP_LOG(DCL_INFO, "profile start done");
        }
        return 0;
    }
}