//
// Created  on 22-9-20.
//

#ifndef DCL_WRAPPER_RESNET_H
#define DCL_WRAPPER_RESNET_H

#include "base/base_classification.h"

namespace dcl {
    class ResNet : public BaseClassifier {
    public:
        int preprocess(std::vector<dcl::Mat> &images) override { return 0; };
    };
}
#endif //DCL_WRAPPER_RESNET_H
