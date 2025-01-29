cmake_minimum_required(VERSION 3.16)
project(LabelReaderProject)

# 使用 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加源文件
add_executable(label-reader
        text-file/read_labels.cpp
        src/read_dataset_labels.cpp
)

# 添加包含路径
include_directories(${CMAKE_SOURCE_DIR}/src)
