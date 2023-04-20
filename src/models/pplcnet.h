//
// Created by intellif on 23-4-19.
//

#ifndef DCL_WRAPPER_PPLCNET_H
#define DCL_WRAPPER_PPLCNET_H

#include "base/base_classification.h"
#include <string>
#include <vector>

namespace dcl {
    static const std::vector<std::string> ageList = {"AgeLess18", "Age18-60", "AgeOver60"};
    static const std::vector<std::string> directList = {"Front", "Side", "Back"};
    static const std::vector<std::string> bagList = {"HandBag", "ShoulderBag", "Backpack"};
    static const std::vector<std::string> upperList = {"UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice"};
    static const std::vector<std::string> lowerList = {"LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress"};

    class PPLCNet : public BaseClassifier {
    public:
        int preprocess(const std::vector<dcl::Mat> &images) override;

        int postprocess(const std::vector<dcl::Mat> &images, std::vector<classification_t> &outputs) override;
    private:
        float threshold_{0.5f};
        float glasses_threshold_{0.3f};
        float hold_threshold_{0.6f};
    };
}

#endif //DCL_WRAPPER_PPLCNET_H
