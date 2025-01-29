echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."
#!/bin/bash

# 定义输入文件和输出文件
input_file="path/to/dataset/images.bin"
output_file="out-tensor-single-image.txt"

# 使用读取工具将二进制图像据转换为张量，并以文本形式输出
./read_dataset_images "$input_file" "$output_file"

echo "图像数据已成功转换为张量，并保存在 $output_file 中。"
