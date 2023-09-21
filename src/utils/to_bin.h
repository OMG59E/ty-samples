//
// Created by xingwg on 8/4/23.
//

#ifndef DCL_WRAPPER_TO_BIN_H
#define DCL_WRAPPER_TO_BIN_H

#include <fstream>
#include <string>


static void to_bin(void* data, size_t size, const char* filename) {
    std::ofstream fs(filename, std::ios::binary);
    std::string str = std::string(static_cast<char*>(data), size);
    fs << str;
    fs.close();
}

#endif //DCL_WRAPPER_TO_BIN_H
