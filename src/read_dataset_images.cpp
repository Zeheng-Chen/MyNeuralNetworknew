//
// Created by msn07 on 2025/1/10.
//

#include "read_dataset_images.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

DatasetImageReader::DatasetImageReader(const std::string& file_path) : file_path_(file_path) {}

Tensor<float> DatasetImageReader::read_image() {
    std::ifstream file(file_path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image file: " + file_path_);
    }

    // 每个图像 28x28
    Tensor<float> image({1, 28, 28});
    for (int i = 0; i < 28 * 28; ++i) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        image.data()[i] = static_cast<float>(pixel) / 255.0f;  // 归一化
    }
    return image;
}

void DatasetImageReader::pretty_print(const Tensor<float>& tensor, const std::string& output_file) {
    std::ofstream out(output_file);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    const auto& data = tensor.data();
    for (size_t i = 0; i < data.size(); ++i) {
        out << std::fixed << std::setprecision(6) << data[i];
        if ((i + 1) % 28 == 0) {
            out << "\n";
        } else {
            out << " ";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_file> <output_file>\n";
        return 1;
    }

    std::string image_file = argv[1];
    std::string output_file = argv[2];

    try {
        DatasetImageReader reader(image_file);
        Tensor<float> image = reader.read_image();
        reader.pretty_print(image, output_file);
        std::cout << "Image successfully processed and saved to " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
