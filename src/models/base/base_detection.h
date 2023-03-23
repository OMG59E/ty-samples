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
        virtual int load(const std::string &modelPath, bool enableAipp);

        /**
         * preprocess
         * @param images
         * @return
         */
        virtual int preprocess(std::vector<dcl::Mat> &images) = 0;

        /**
         * model inference for signal input
         * @param image
         * @param detections
         * @return
         */
        virtual int inference(dcl::Mat &image, std::vector<dcl::detection_t> &detections);

        /**
         * model inference for multi-input
         * @param images
         * @param detections
         * @return
         */
        virtual int inference(std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections);

        /**
         * postprocess
         * @param images
         * @param detections
         * @return
         */
        virtual int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::detection_t> &detections) = 0;

        /**
         * unload model
         * @return
         */
        virtual int unload() ;

    protected:
        std::vector<dcl::Tensor> vOutputTensors_;
        NetOperator net_;
    };
}
#endif //DCL_WRAPPER_BASE_DETECTION_H
