#!/usr/bin/env bash

# shellcheck disable=SC2164
mkdir -p build_x86_64 && cd build_x86_64
cmake -DCMAKE_BUILD_TYPE=RELEASE -DHOST_TYPE=x86 ..
make
