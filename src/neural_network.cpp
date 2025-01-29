//
// Created by msn07 on 2025/1/10.
//
#include "neural_network.hpp"
#include <iostream>
#include <algorithm>

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
