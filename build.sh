#!/usr/bin/env bash

# shellcheck disable=SC2164
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DHOST_TYPE=a55 ..
make
