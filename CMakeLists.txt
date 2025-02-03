#cmake_minimum_required(VERSION 3.16)
#project(LabelReaderProject)
#
## 使用 C++20 标准
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#
## 添加源文件
#add_executable(label-reader
#        text-file/read_labels.cpp
#        src/read_dataset_labels.cpp
#)
#
#
## 添加包含路径
#include_directories(${CMAKE_SOURCE_DIR}/src)
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MyNeuralNetwork)

# 启用 C++11 或更高标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找所有源文件
set(SRC_DIR src)
set(SOURCES
        ${SRC_DIR}/neural_network.cpp
        ${SRC_DIR}/read_dataset_images.cpp
        ${SRC_DIR}/read_dataset_labels.cpp
)

# 生成可执行文件
add_executable(read_dataset_images ${SRC_DIR}/read_dataset_images.cpp)
add_executable(read_dataset_labels ${SRC_DIR}/read_dataset_labels.cpp)
add_executable(neural_network ${SRC_DIR}/neural_network.cpp)

# 链接可能需要的库 (如果有)
# target_link_libraries(neural_network some_library)
