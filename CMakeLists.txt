cmake_minimum_required(VERSION 2.8.12)
project(dcl_wrapper C CXX)

set(DENGINE_ROOT /DEngine)
set(HCP_PATH ${DENGINE_ROOT}/tyhcp)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-O0 -g -std=c++11 -w -fPIC")
else()
    set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -w -fPIC")
endif()

set(HCP_LIB_PATH ${HCP_PATH}/a55/aarch64-linux-gcc9.2.1/lib)
set(DPF_LIB_PATH ${DENGINE_ROOT}/tydpf/a55/aarch64-linux-gcc9.2.1/lib)
set(OPENCV_PATH /mnt/modelzoo/3rdparty/opencv)

include_directories(${HCP_PATH}/include/dcl)
include_directories(src/graph/include)
include_directories(src/common)
include_directories(src/utils)
include_directories(src)
include_directories(${OPENCV_PATH}/include)
include_directories(${OPENCV_PATH}/include/opencv4)

link_directories(${HCP_LIB_PATH})
link_directories(${DPF_LIB_PATH})
link_directories(${OPENCV_PATH}/lib)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)

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
