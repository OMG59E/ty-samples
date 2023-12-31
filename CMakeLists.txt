cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0048 NEW)
project(dcl_wrapper C CXX)

set(HCP_INC_PATH /app/include/hcp)
set(HAL_INC_PATH /app/include/hal)
set(MAL_INC_PATH /app/include/mal)
set(HCP_LIB_PATH /app/lib/hcp)
set(HAL_LIB_PATH /app/lib/hal)
set(MAL_LIB_PATH /app/lib/mal)
set(OPENCV_PATH 3rdparty/opencv)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-O0 -g -std=c++14 -w -fPIC")
else()
    set(CMAKE_CXX_FLAGS "-O2 -std=c++14 -w -fPIC")
endif()

include_directories(${HCP_INC_PATH})
include_directories(${HAL_INC_PATH})
include_directories(${MAL_INC_PATH})
include_directories(${OPENCV_PATH}/include)
include_directories(${OPENCV_PATH}/include/opencv4)

include_directories(src/common)
include_directories(src/utils)
include_directories(src)

link_directories(${HCP_LIB_PATH})
link_directories(${HAL_LIB_PATH})
link_directories(${MAL_LIB_PATH})
link_directories(${OPENCV_PATH}/lib)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(3rdparty/pybind11)

list(APPEND SDK_LIBS ty_dcl)
list(APPEND SDK_LIBS ty_common)
list(APPEND SDK_LIBS ty_drv)
list(APPEND SDK_LIBS ty_drv_memory)
list(APPEND SDK_LIBS ty_ge)
list(APPEND SDK_LIBS ty_stream_sched)
list(APPEND SDK_LIBS ty_syslink)
list(APPEND SDK_LIBS ty_utils)
list(APPEND SDK_LIBS ty_dsmi_device)
list(APPEND SDK_LIBS hcp_hal)
list(APPEND SDK_LIBS hal_device)
list(APPEND SDK_LIBS ipcm)
list(APPEND SDK_LIBS mal)
list(APPEND OPENCV_LIBS opencv_core)
list(APPEND OPENCV_LIBS opencv_imgproc)
list(APPEND OPENCV_LIBS opencv_imgcodecs)

add_library(${PROJECT_NAME} SHARED ${DCL_SOURCES})
target_link_libraries(${PROJECT_NAME} ${SDK_LIBS} ${OPENCV_LIBS} pthread dl)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
pybind11_add_module(pymz SHARED python/pymz.cpp)
target_link_libraries(pymz PUBLIC ${PROJECT_NAME})

foreach(EXAMPLE_NAME ${EXAMPLE_SOURCES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${EXAMPLE_NAME})
    list(GET SEXY_LIST -1 TEST_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
    list(GET SEXY_LIST 0 TEST_CASE_NAME)
    add_executable(${TEST_CASE_NAME} ${EXAMPLE_NAME})
    target_link_libraries(${TEST_CASE_NAME} ${PROJECT_NAME})
    set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
endforeach()

foreach(TOOL_NAME ${TOOL_SOURCES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${TOOL_NAME})
    list(GET SEXY_LIST -1 TOOL_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TOOL_CASE_NAME})
    list(GET SEXY_LIST 0 TOOL_CASE_NAME)
    add_executable(${TOOL_CASE_NAME} ${TOOL_NAME})
    target_link_libraries(${TOOL_CASE_NAME} ${PROJECT_NAME})
    set_target_properties(${TOOL_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
endforeach()
