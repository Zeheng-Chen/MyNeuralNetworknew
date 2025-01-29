//
// Created by msn07 on 2025/1/8.
//

// read_dataset_labels.cpp

#include "read_dataset_labels.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

// DatasetLabelReader 构造函数
DatasetLabelReader::DatasetLabelReader(const std::string& filePath) : filePath(filePath) {
    // 您可以在此处添加初始化逻辑或调试信息（如需要）
}


Tensor<int> DatasetLabelReader::readLabels() const {


    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    std::vector<int> labels;
    std::string line;

    // 逐行读取标签
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        int label;
        while (lineStream >> label) {
            labels.push_back(label);
        }
    }
    file.close();

    // 确定张量的形状（1D 张量）
    std::vector<size_t> shape = {labels.size()};
    Tensor<int> tensor(shape); // 初始化张量

    // 将标签填充到张量中
    for (size_t i = 0; i < labels.size(); ++i) {
        tensor({i}) = labels[i]; // 使用索引设置张量值
    }

    return tensor;
}


