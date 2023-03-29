//
// Created  on 22-8-24.
//
#include "net_operator.h"
#include "utils/utils.h"
#include "utils/macro.h"

namespace dcl {
    int NetOperator::load(const std::string &modelPath, bool enableAipp) {
        // void *mdlData{nullptr};
        // uint32_t mdlSize;
        // readBinFile(modelPath, mdlData, mdlSize);
        // dclError ret = dclmdlQuerySizeFromMem(mdlData, mdlSize, &workSize_, &weightSize_);
        // if (DCL_SUCCESS != ret) {
        //     DCL_APP_LOG(DCL_ERROR, "query model failed, model file is %s, errorCode is %d",
        //                 modelPath.c_str(), static_cast<int32_t>(ret));
        //     return -1;
        // }
        // ret = dclmdlLoadFromMem(mdlData, mdlSize, &id_);
        // if (DCL_SUCCESS != ret) {
        //     DCL_APP_LOG(DCL_ERROR, "load model from file failed, model file is %s, errorCode is %d",
        //                 modelPath.c_str(), static_cast<int32_t>(ret));
        //     return -2;
        // }
        // DCLRT_FREE(mdlData);

        dclError ret = dclmdlLoadFromFile(modelPath.c_str(), &id_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "Failed to load model, model file is %s, errorCode is %d",
                        modelPath.c_str(), static_cast<int32_t>(ret));
            return -1;
        }

        loadFlag_ = true;
        DCL_APP_LOG(DCL_INFO, "load model %s success", modelPath.c_str());

        ret = createModelDesc();
        if (ret != 0)
            return -5;

        if (createInputDataset() != 0) {
            DCL_APP_LOG(DCL_ERROR, "Failed to createInputDataset, modelId %d", id_);
            return -6;
        }

        if (createOutputDataset() != 0) {
            DCL_APP_LOG(DCL_ERROR, "Failed to createOutputDataset, modelId %d", id_);
            return -7;
        }

        return 0;
    }

    int NetOperator::unload() {
        if (!loadFlag_)
            return 0;

        dclError ret = dclmdlUnload(id_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "unload model failed, modelId is %u, errorCode is %d",
                        id_, static_cast<int32_t>(ret));
            return -1;
        }

        destroyInputDataset();
        for (auto& input : vInputs_) {
            DCLRT_FREE(input.aippData);
            DCLRT_FREE(input.data);
        }

        destroyOutputDataset();

        destroyModelDesc();
        DCLRT_FREE(workPtr_);
        DCLRT_FREE(weightPtr_);
        workSize_ = 0;
        weightSize_ = 0;

        DCL_APP_LOG(DCL_INFO, "unload model success, modelId is %u", id_);
        id_ = 0;
        loadFlag_ = false;

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

        dclError ret = dclmdlGetDesc(desc_, id_);
        if (ret != DCL_ERROR_NONE) {
            DCL_APP_LOG(DCL_ERROR, "get model description failed, modelId is %u, errorCode is %d",
                        id_, static_cast<int32_t>(ret));
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
        if (inputDataset_)
            updateDataBuffer();
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        dclError ret = dclmdlExecute(id_, inputDataset_, outputDataset_);
        if (DCL_SUCCESS != ret) {
            DCL_APP_LOG(DCL_ERROR, "Failed to execute model, modelId is %u, errorCode is %d",
                        id_, static_cast<int32_t>(ret));
            return -3;
        }

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        vOutputTensors.clear();
        vOutputTensors.resize(nbNumOutput_);
        for (int i = 0; i < nbNumOutput_; ++i) {
            dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(outputDataset_, i);
            void *data = dclGetDataBufferAddr(dataBuffer);
            vOutputTensors[i].data = reinterpret_cast<float *>(data);
            vOutputTensors[i].nbDims = int(vOutputDims_[i].dimCount);
            for (int j = 0; j < vOutputTensors[i].nbDims; ++j)
                vOutputTensors[i].d[j] = int(vOutputDims_[i].dims[j]);
            dclFormat format = dclmdlGetOutputFormat(desc_, i);
            dclDataType datatype = dclmdlGetOutputDataType(desc_, i);
            if (DCL_FLOAT16 == datatype) {
                // TODO fp16 -> fp32

            } else if (DCL_FLOAT == datatype) {
                // nothing
            } else {
                DCL_APP_LOG(DCL_ERROR, "Not support data type %d", datatype);
                return -5;
            }

            if (DCL_FORMAT_NCHW == format || DCL_FORMAT_UNDEFINED == format)
                vOutputTensors[i].dataLayout = DCL_FORMAT_NCHW;
            else if (DCL_FORMAT_NHWC == format)
                vOutputTensors[i].dataLayout = DCL_FORMAT_NHWC;
            else {
                DCL_APP_LOG(DCL_ERROR, "Not support format %d", format);
                return -4;
            }
        }
        high_resolution_clock::time_point t3 = high_resolution_clock::now();
        duration<float, std::micro> tp0 = t1 - t0;
        duration<float, std::micro> tp1 = t2 - t1;
        duration<float, std::micro> tp2 = t3 - t2;
        DCL_APP_LOG(DCL_INFO, "UpdateDataBuffer: %.3fms, Execute: %.3fms, GetOutput: %.3fms",
                    tp0.count() / 1000.0f, tp1.count() / 1000.0f, tp2.count() / 1000.0f);
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
            // dclmdlInputAippType type{};
            // size_t dynamicAttachedDataIndex;
            // ret = dclmdlGetAippType(id_, n, &type, &dynamicAttachedDataIndex);
            // if (DCL_SUCCESS != ret) {
            //     DCL_APP_LOG(DCL_ERROR, "Failed to get aipp type, error code: %d", static_cast<int32_t>(ret));
            //     return -1;
            // }

            // if (DCL_DYNAMIC_AIPP_NODE == type) {
            //     nbAippInput_++;
            //     dclDataBuffer *aippBuffer = dclCreateDataBuffer(nullptr, 0);
            //     ret = dclmdlAddDatasetBuffer(inputDataset_, aippBuffer);
            //     if (DCL_SUCCESS != ret) {
            //         DCL_APP_LOG(DCL_ERROR, "Failed to add input dataset buffer, error code: %d",
            //                     static_cast<int32_t>(ret));
            //         DCLMDL_DATABUFFER_FREE(aippBuffer);
            //         return -2;
            //     }
            //     continue;
            // }

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
            // input.aippIdx = DCL_DATA_WITHOUT_AIPP == type ? -1 : int(dynamicAttachedDataIndex);
            dclmdlIODims dims;
            memset(&dims, 0, sizeof(dims));
            if (!input.hasAipp()) {
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

                // ret = dclrtMalloc(&(input.data), input.dataSize*sizeof(uint8_t), DCL_MEM_MALLOC_NORMAL_ONLY);
                ret = dclrtMallocEx(&(input.data), &(input.phyaddr), input.dataSize*sizeof(uint8_t), 16, DCL_MEM_MALLOC_NORMAL_ONLY);
                if (DCL_SUCCESS != ret) {
                    DCL_APP_LOG(DCL_ERROR, "Failed to malloc data buffer without aipp, error code: %d", ret);
                    return -5;
                }

            } else {
                //ret = dclrtMalloc(&(input.data), MAX_IMAGE_SIZE*sizeof(uint8_t), DCL_MEM_MALLOC_NORMAL_ONLY);
                //if (DCL_SUCCESS != ret) {
                //    DCL_APP_LOG(DCL_ERROR, "Failed to malloc data buffer with aipp, error code: %d", ret);
                //    return -6;
                //}
                //input.aippSize = dclmdlGetInputSizeByIndex(desc_, dynamicAttachedDataIndex);
                //ret = dclrtMalloc(&(input.aippData), input.aippSize, DCL_MEM_MALLOC_NORMAL_ONLY);
                //if (DCL_SUCCESS != ret) {
                //    DCL_APP_LOG(DCL_ERROR, "Failed to malloc aipp buffer, error code: %d", ret);
                //    return -7;
                //}
                //DCL_APP_LOG(DCL_INFO, "create data input[%d] success, and aipp buffer size: %d", n, input.aippSize);
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

    int NetOperator::updateDataBuffer() {
        if (!inputDataset_) {
            DCL_APP_LOG(DCL_ERROR, "inputDataset is null");
            return -1;
        }

        dclError ret;
        for (auto input : vInputs_) {
            if (!input.hasAipp()) // no aipp
                continue;

            dclDataBuffer* dataBuffer = dclmdlGetDatasetBuffer(inputDataset_, input.idx);
            ret = dclUpdateDataBuffer(dataBuffer, input.data, input.dataSize);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to update data buffer with aipp, error code: %d", ret);
                return -3;
            }

            // aipp
            dclmdlAIPP *aipp = dclmdlCreateAIPP();
            dclmdlSetAIPPInputFormat(aipp, input.pixelFormat);
            dclmdlSetAIPPSrcImageSize(aipp, input.w(), input.h());
            bool cropSwitch = (input.original_width != input.w()) || (input.original_height != input.h());
            dclmdlSetAIPPCropParams(aipp, (int8_t)cropSwitch, 0, 0, input.original_width, input.original_height);
            ret = dclmdlSetAIPPByInputIndex(id_, inputDataset_, input.idx, aipp);
            if (DCL_SUCCESS != ret) {
                DCL_APP_LOG(DCL_ERROR, "Failed to set aipp[%d] to input[%d], and error code: %d", input.aippIdx, input.idx, ret);
                return -4;
            }
            DCL_APP_LOG(DCL_INFO, "Update data buffer[%d] success", input.idx);
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

    int NetOperator::findAippDataIdx(int aippIdx) const {
        for (const auto& input : vInputs_) {
            if (aippIdx == input.aippIdx)
                return input.idx;
        }
        return -1;
    }
}