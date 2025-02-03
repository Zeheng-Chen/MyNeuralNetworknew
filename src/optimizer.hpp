//
// Created by msn07 on 2025/1/10.
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"

class Optimizer {
public:
    virtual void updateWeights(Tensor<float>& weights, const Tensor<float>& gradients) = 0;
    virtual ~Optimizer() = default;
    virtual float getLearningRate() const = 0;
};

// SGD 优化器
class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(float learningRate);
    void updateWeights(Tensor<float>& weights, const Tensor<float>& gradients) override;
    float getLearningRate() const { return learningRate; }


private:
    float learningRate;
};

// SGD 优化器实现
inline SGDOptimizer::SGDOptimizer(float learningRate) : learningRate(learningRate) {}

inline void SGDOptimizer::updateWeights(Tensor<float>& weights, const Tensor<float>& gradients) {
    for (size_t i = 0; i < weights.shape()[0]; ++i) {
        weights.data()[i] -= learningRate * gradients.data()[i];
    }
}

#endif // OPTIMIZER_HPP
