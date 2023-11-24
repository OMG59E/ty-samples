//
// Created by xingwg on 11/24/23.
//
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <utility>
#include <json/json.h>
#include "macro.h"


PYBIND11_MODULE(pymz, m) {
    m.doc() = "Python bindings for tyhcp sdk";
}
