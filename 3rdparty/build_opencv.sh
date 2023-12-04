#!/usr/bin/env bash

# shellcheck disable=SC2164
cd opencv-4.5.4 && mkdir -p build && cd build

cmake \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=../../opencv \
-DCMAKE_CXX_COMPILER="${ARM_CXX_COMPILER}" \
-DCMAKE_C_COMPILER="${ARM_C_COMPILER}" \
-DCMAKE_CXX_FLAGS="-O2 -std=c++11 -w -fPIC ${ARM_CXX_FLAGS}" \
-DWITH_CUDA=OFF \
-DWITH_OPENCL=OFF \
-DWITH_FFMPEG=OFF \
-DWITH_1394=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_IMGCODEC_PXM=OFF \
-DWITH_IMGCODEC_HDR=OFF \
-DWITH_JPEG=ON \
-DWITH_JPEG2000=ON \
-DWITH_IMGCODEC_SUNRASTER=OFF \
-DWITH_GTK=OFF \
-DWITH_GSTREAMER=OFF \
-DWITH_GPHOTO2=OFF \
-DWITH_MATLAB=OFF \
-DWITH_PNG=ON \
-DWITH_VTK=OFF \
-DWITH_WEBP=OFF \
-DWITH_TIFF=OFF \
-DWITH_IPP=OFF \
-DWITH_V4L=OFF \
-DWITH_PROTOBUF=OFF \
-DBUILD_opencv_python3=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_ZLIB=ON \
-DBUILD_IPP_IW=OFF \
-DBUILD_ITT=OFF \
-DBUILD_JAVA=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_PROTOBUF=OFF \
-DBUILD_PACKAGE=OFF \
-DBUILD_opencv_calib3d=OFF \
-DBUILD_opencv_dnn=OFF \
-DBUILD_opencv_features2d=OFF \
-DBUILD_opencv_flann=OFF \
-DBUILD_opencv_highgui=OFF \
-DBUILD_opencv_ml=OFF \
-DBUILD_opencv_objdetect=OFF \
-DBUILD_opencv_photo=OFF \
-DBUILD_opencv_shape=OFF \
-DBUILD_opencv_stitching=OFF \
-DBUILD_opencv_superres=OFF \
-DBUILD_opencv_ts=OFF \
-DBUILD_opencv_video=OFF \
-DBUILD_opencv_gapi=OFF \
-DBUILD_opencv_videoio=OFF \
-DBUILD_opencv_videostab=OFF \
-DBUILD_opencv_java_bindings_generator=OFF \
-DBUILD_opencv_python_bindings_generator=OFF \
-DBUILD_opencv_apps=OFF ..

make -j20 && make install
