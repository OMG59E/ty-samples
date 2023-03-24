//
// Created  on 22-8-25.
//

#ifndef DCL_WRAPPER_BASE_DETECTION_H
#define DCL_WRAPPER_BASE_DETECTION_H

#include "base_type.h"
#include "net_operator.h"

namespace dcl {
    class BaseDetector {
    public:
        /**
         * load model from file
         * @param modelPath
         * @param enableAipp
         * @return
         */
        virtual int load(const std::string &modelPath);

        /**
         * preprocess
         * @param images
         * @return
         */
        virtual int preprocess(const std::vector<dcl::Mat> &images);

        /**
         * model inference for signal input
         * @param image
         * @param outputs
         * @return
         */
        virtual int inference(const dcl::Mat &image, std::vector<detection_t> &outputs);

        /**
         * model inference for multi-input
         * @param images
         * @param detections
         * @return
         */
        virtual int inference(const std::vector<dcl::Mat> &images, std::vector<detection_t> &outputs);

        /**
         * postprocess
         * @param images
         * @param detections
         * @return
         */
        virtual int postprocess(const std::vector<dcl::Mat> &images, std::vector<detection_t> &outputs) = 0;

        /**
         * unload model
         * @return
         */
        virtual int unload();

    protected:
        std::vector<dcl::Tensor> vOutputTensors_;
        NetOperator net_;
    };
}
#endif //DCL_WRAPPER_BASE_DETECTION_H
