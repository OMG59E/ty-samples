//
// Created by intellif on 23-4-14.
//

#ifndef DCL_WRAPPER_MATH_UTILS_H
#define DCL_WRAPPER_MATH_UTILS_H

#include <cmath>

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

#endif //DCL_WRAPPER_MATH_UTILS_H
