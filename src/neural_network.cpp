//
// Created by msn07 on 2025/1/10.
//
#include "neural_network.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>

NeuralNetwork::NeuralNetwork() : lossFunction(nullptr), optimizer(nullptr) {}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void NeuralNetwork::setLoss(LossFunction* loss) {
    lossFunction = loss;
}

void NeuralNetwork::setOptimizer(Optimizer* optimizer) {
    this->optimizer = optimizer;
}

Tensor<float> NeuralNetwork::forward(const Tensor<float>& input) {
    Tensor<float> output = input;
    for (Layer* layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Tensor<float>& target) {
    Tensor<float> grad = lossFunction->computeGradient(layers.back()->getOutput(), target);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void NeuralNetwork::updateWeights() {
    for (Layer* layer : layers) {
        layer->updateWeights(optimizer->getLearningRate());
    }
}

void NeuralNetwork::train(const std::vector<Tensor<float>>& images,
                          const std::vector<Tensor<float>>& labels,
                          int epochs, int batchSize) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        for (size_t i = 0; i < images.size(); i += batchSize) {
            // Batch processing
            Tensor<float> batchInput = images[i];
            Tensor<float> batchLabel = labels[i];

            // Forward pass
            Tensor<float> predictions = forward(batchInput);

            // Compute loss
            float loss = lossFunction->computeLoss(predictions, batchLabel);
            totalLoss += loss;

            // Backward pass
            backward(batchLabel);

            // Update weights
            updateWeights();
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << totalLoss / images.size() << std::endl;
    }
}

int NeuralNetwork::predict(const Tensor<float>& input) {
    Tensor<float> output = forward(input);
    auto maxElementIt = std::max_element(output.data().begin(), output.data().end());
    return std::distance(output.data().begin(), maxElementIt);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_file> <label_file> <epochs>\n";
        return 1;
    }

    std::string image_file = argv[1];
    std::string label_file = argv[2];
    int epochs = std::stoi(argv[3]);

    try {
        // 加载数据
        std::vector<Tensor<float>> images;
        std::vector<Tensor<float>> labels;

        // 读取图像数据
        std::ifstream img_stream(image_file);
        if (!img_stream) {
            throw std::runtime_error("Failed to open image file: " + image_file);
        }
        Tensor<float> img_tensor({1, 28, 28});
        for (float& val : img_tensor.data()) {
            img_stream >> val;
        }
        images.push_back(img_tensor);

        // // 读取标签数据
        // std::ifstream lbl_stream(label_file);
        // if (!lbl_stream) {
        //     throw std::runtime_error("Failed to open label file: " + label_file);
        // }
        // Tensor<int> lbl_tensor({1});
        // lbl_stream >> lbl_tensor.data()[0];
        // labels.push_back(lbl_tensor);
        // 读取标签数据
        std::ifstream lbl_stream(label_file);
        if (!lbl_stream) {
            throw std::runtime_error("Failed to open label file: " + label_file);
        }
        Tensor<int> lbl_tensor({1});
        lbl_stream >> lbl_tensor.data()[0];

        // 转换 Tensor<int> 为 Tensor<float>
        Tensor<float> label_float(lbl_tensor.shape());
        for (size_t i = 0; i < lbl_tensor.shape()[0]; ++i) {
            label_float.data()[i] = static_cast<float>(lbl_tensor.data()[i]);
        }

        // 存入 labels
        labels.push_back(label_float);


        // 初始化神经网络
        NeuralNetwork network;

        // 设置损失函数和优化器（假设已定义）
        LossFunction* loss = nullptr;  // 这里应初始化你的损失函数
        Optimizer* optimizer = nullptr;  // 这里应初始化你的优化器
        network.setLoss(loss);
        network.setOptimizer(optimizer);

        // 训练神经网络
        network.train(images, labels, epochs, 1);

        // 进行预测
        int prediction = network.predict(images[0]);
        std::cout << "Predicted label: " << prediction << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
