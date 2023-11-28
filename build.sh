#!/usr/bin/env bash

# shellcheck disable=SC2164
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DPYTHON_EXECUTABLE=/usr/bin/python3.8 ..
make -j4
