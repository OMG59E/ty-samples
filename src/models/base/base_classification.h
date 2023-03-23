//
// Created  on 22-9-20.
//

#ifndef DCL_WRAPPER_BASE_CLASSIFICATION_H
#define DCL_WRAPPER_BASE_CLASSIFICATION_H
#include "base_type.h"
#include "net_operator.h"

namespace dcl {
    class BaseClassifier {
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
        virtual int inference(dcl::Mat &image, std::vector<dcl::classification_t> &classifications);

        /**
         * model inference for multi-input
         * @param images
         * @param detections
         * @return
         */
        virtual int inference(std::vector<dcl::Mat> &images, std::vector<dcl::classification_t> &classifications);

        /**
         * postprocess
         * @param images
         * @param detections
         * @return
         */
        virtual int postprocess(const std::vector<dcl::Mat> &images, std::vector<dcl::classification_t> &classifications);

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
#endif //DCL_WRAPPER_BASE_CLASSIFICATION_H
