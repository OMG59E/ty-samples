//
// Created  on 22-8-25.
//

#ifndef DCL_WRAPPER_BASE_DETECTION_H
#define DCL_WRAPPER_BASE_DETECTION_H

#include "base_type.h"
#include "net_operator.h"

namespace ty {
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
        virtual int preprocess(const std::vector<ty::Mat> &images);

        /**
         * model inference for signal input
         * @param image
         * @param outputs
         * @return
         */
        virtual int inference(const ty::Mat &image, std::vector<detection_t> &outputs);

        /**
         * model inference for multi-input
         * @param images
         * @param detections
         * @return
         */
        virtual int inference(const std::vector<ty::Mat> &images, std::vector<detection_t> &outputs);

        /**
         * postprocess
         * @param images
         * @param detections
         * @return
         */
        virtual int postprocess(const std::vector<ty::Mat> &images, std::vector<detection_t> &outputs) = 0;

        /**
         * unload model
         * @return
         */
        virtual int unload();

        virtual void set_iou_threshold(float iou_threshold)  { iou_threshold_ = iou_threshold; }
        virtual void set_conf_threshold(float conf_threshold) { conf_threshold_ = conf_threshold; }

    protected:
        std::vector<ty::Tensor> vOutputTensors_;
        NetOperator net_;
        float iou_threshold_{0.45f};
        float conf_threshold_{0.25f};
    };
}
#endif //DCL_WRAPPER_BASE_DETECTION_H
