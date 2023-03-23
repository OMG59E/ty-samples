//
// Created  on 22-8-24.
//

#ifndef DCL_WRAPPER_MACRO_H
#define DCL_WRAPPER_MACRO_H

#include "dcl.h"

#define DCLMDL_DESC_FREE(ptr)              \
do {                                       \
    if (ptr) {                             \
        (void) dclmdlDestroyDesc(ptr);     \
        ptr = nullptr;                     \
    }                                      \
} while(0)


#define DCLMDL_DATASET_FREE(ptr)           \
do {                                       \
    if (ptr) {                             \
        (void) dclmdlDestroyDataset(ptr);  \
        ptr = nullptr;                     \
    }                                      \
} while(0)


#define DCLMDL_DATABUFFER_FREE(ptr)        \
do {                                       \
    if (ptr) {                             \
        (void) dclDestroyDataBuffer(ptr);  \
        ptr = nullptr;                     \
    }                                      \
} while(0)


#define DCLRT_FREE(ptr)                    \
do {                                       \
    if (ptr) {                             \
        (void) dclrtFree(ptr);             \
        ptr = nullptr;                     \
    }                                      \
} while(0)


#define DCL_PROFILE_FREE(ptr)              \
do {                                       \
    if (ptr) {                             \
        dclprofDestroyConfig(ptr);         \
        ptr = nullptr;                     \
    }                                      \
} while(0)
#endif //DCL_WRAPPER_MACRO_H
