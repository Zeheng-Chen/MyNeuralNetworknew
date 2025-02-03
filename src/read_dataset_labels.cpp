//
// Created by msn07 on 2025/1/8.
//

// read_dataset_labels.cpp


#include <iostream>
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


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <label_file> <output_file>\n";
        return 1;
    }

    std::string label_file = argv[1];
    std::string output_file = argv[2];

    try {
        DatasetLabelReader reader(label_file);
        Tensor<int> labels = reader.readLabels();

        // 将标签写入输出文件
        std::ofstream out(output_file);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }

        for (const auto& value : labels.data()) {
            out << value << "\n";
        }

        std::cout << "Labels successfully processed and saved to " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
