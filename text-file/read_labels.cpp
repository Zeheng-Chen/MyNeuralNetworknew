//
// Created by msn07 on 2025/1/8.
//
#include "../src/read_dataset_labels.hpp"
#include <iostream>

int main()
{
    try
    {
        DatasetLabelReader reader("/mnt/d/studieren/advanced Programming Techniques/ws2024-group-20-gile/expected-results/out-tensor-single-label.txt");
        Tensor<int> labels = reader.readLabels();
        std::cout << "Labels loaded successfully." << std::endl;
        //labels.print(); // 假设 Tensor 类有 print 方法
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
