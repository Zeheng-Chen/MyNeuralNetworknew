//
// Created by msn07 on 2025/1/8.
//

// read_dataset_labels.hpp

#ifndef READ_DATASET_LABELS_HPP
#define READ_DATASET_LABELS_HPP

#include "tensor.hpp"
#include <string>

class DatasetLabelReader
{
public:
    // Constructor
    DatasetLabelReader(const std::string &filePath);

    // Reads the dataset labels into a Tensor
    Tensor<int> readLabels() const;

private:
    std::string filePath;
};

#endif // READ_DATASET_LABELS_HPP
