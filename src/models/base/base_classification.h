//
// Created  on 22-9-20.
//

#ifndef DCL_WRAPPER_BASE_CLASSIFICATION_H
#define DCL_WRAPPER_BASE_CLASSIFICATION_H

#include "base_type.h"
#include "utils/color.h"
#include "utils/resize.h"
#include "net_operator.h"

namespace ty {
    class BaseClassifier {
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
        virtual int preprocess(const std::vector<ty::Mat> &images);

        /**
         * model inference for signal input
         * @param image
         * @param outputs
         * @return
         */
        virtual int inference(const ty::Mat &image, std::vector<classification_t> &outputs);

        /**
         * model inference for multi-input
         * @param images
         * @param detections
         * @return
         */
        virtual int inference(const std::vector<ty::Mat> &images, std::vector<classification_t> &outputs);

        /**
         * postprocess
         * @param images
         * @param detections
         * @return
         */
        virtual int postprocess(const std::vector<ty::Mat> &images, std::vector<classification_t> &outputs);

        /**
         * unload model
         * @return
         */
        virtual int unload();

    protected:
        std::vector<ty::Tensor> vOutputTensors_;
        NetOperator net_;
    };
}
#endif //DCL_WRAPPER_BASE_CLASSIFICATION_H
