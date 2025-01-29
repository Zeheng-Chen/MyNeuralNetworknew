//
// Created by msn07 on 2025/1/10.
//

#ifndef READ_DATASET_IMAGES_HPP
#define READ_DATASET_IMAGES_HPP
#include "tensor.hpp"
#include <string>

class DatasetImageReader
{
  public:
    //Constructor
    DatasetImageReader(const std::string& file_path);

    //Reads the dataset labels into tensor
    Tensor<float> read_image(); //read single image as tensor
    void pretty_print(const Tensor<float>& tensor, const std::string& output_file);
private:
    std::string file_path_;
};

#endif //READ_DATASET_IMAGES_HPP
