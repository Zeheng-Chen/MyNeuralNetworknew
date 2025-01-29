echo "This script should trigger the training and testing of your neural network implementation..."
#!/bin/bash

# 定义训练所需的图像和标签路径
image_data="out-tensor-single-image.txt"
label_data="out-tensor-single-label.txt"
log_file="out-prediction-log-single-image.txt"

# 执行神经网络训练和测试
./neural_network_train_test "$image_data" "$label_data" > "$log_file"

echo "MNIST 训练和测试已完成，日志已保存至 $log_file。"
