//
// Created by msn07 on 2025/1/10.
//

#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"
#include<cmath>

// 抽象损失函数基类
class LossFunction {
public:
    virtual float computeLoss(const Tensor<float>& predictions, const Tensor<float>& targets) = 0;
    virtual Tensor<float> computeGradient(const Tensor<float>& predictions, const Tensor<float>& targets) = 0;
    virtual ~LossFunction() = default;
};

// 交叉熵损失
class CrossEntropyLoss : public LossFunction {
public:
    float computeLoss(const Tensor<float>& predictions, const Tensor<float>& targets) override;
    Tensor<float> computeGradient(const Tensor<float>& predictions, const Tensor<float>& targets) override;
};

// 交叉熵损失实现
inline float CrossEntropyLoss::computeLoss(const Tensor<float>& predictions, const Tensor<float>& targets) {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.shape()[0]; ++i) {
        loss -= targets.data()[i] * std::log(predictions.data()[i] + 1e-9f); // 加上一个小值避免log(0)
    }
    return loss / targets.shape()[0];
}

inline Tensor<float> CrossEntropyLoss::computeGradient(const Tensor<float>& predictions, const Tensor<float>& targets) {
    Tensor<float> gradients(predictions.shape());
    for (size_t i = 0; i < predictions.shape()[0]; ++i) {
        gradients.data()[i] = (predictions.data()[i] - targets.data()[i]) / (predictions.data()[i] * (1.0f - predictions.data()[i]) + 1e-9f);
    }
    return gradients;
}

#endif // LOSS_HPP
