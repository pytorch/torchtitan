# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


cmake_minimum_required(VERSION 3.12)
project(autopipe)

# 使用最简单的方式，避免所有 Modern CMake 特性

# 查找 Python
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

# 获取 Python 扩展名
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 获取 pybind11 包含目录
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 创建模块
add_library(autopipe MODULE autopipe.cpp)

# 设置目标属性
set_target_properties(autopipe PROPERTIES
    PREFIX ""
    SUFFIX ${PYTHON_MODULE_EXTENSION}
    OUTPUT_NAME "autopipe"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 包含目录
include_directories(
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(autopipe ${PYTHON_LIBRARIES})
