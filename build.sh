echo "This script should build your project now..."
#!/bin/bash

# 使用 CMake 和 Make 构建项目
mkdir -p build
cd build
cmake ..
make

echo "项目已成功构建。"
