//
// Created by msn07 on 2025/1/10.
//



#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class Layer {
public:
    virtual Tensor<float> forward(const Tensor<float>& input) = 0;
    virtual Tensor<float> backward(const Tensor<float>& gradOutput) = 0;
    virtual void updateWeights(float learningRate) = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_HPP



