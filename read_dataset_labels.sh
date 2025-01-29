echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."
#!/bin/bash

# 定义输入文件和输出文件
input_file="path/to/dataset/labels.txt"
output_file="out-tensor-single-label.txt"

# 使用读取工具将标签数据转换为张量，并以文本形式输出
./read_dataset_labels "$input_file" "$output_file"

echo "标签数据已成功转换为张量，并保存在 $output_file 中。"
