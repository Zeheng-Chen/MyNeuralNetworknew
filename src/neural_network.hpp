//
// Created by msn07 on 2025/1/10.
//



#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "tensor.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();
    void addLayer(Layer* layer);
    void setLoss(LossFunction* loss);
    void setOptimizer(Optimizer* optimizer);
    Tensor<float> forward(const Tensor<float>& input);
    void backward(const Tensor<float>& target);
    void updateWeights();
    void train(const std::vector<Tensor<float>>& images, const std::vector<Tensor<float>>& labels, int epochs, int batchSize);
    int predict(const Tensor<float>& input);

private:
    std::vector<Layer*> layers;
    LossFunction* lossFunction;
    Optimizer* optimizer;
};

#endif // NEURAL_NETWORK_HPP

