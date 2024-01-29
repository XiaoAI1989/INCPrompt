#!/bin/bash

# 定义一个二维数组，每个子数组包含一组log目录和gpuid
params=(
  "outputs/Imgrlength=4 4"
#  "outputs/Imgrlength=20 20"

)

# 遍历所有的参数组合
for param in "${params[@]}"; do
   # 拆分参数到各自的变量
   IFS=' ' read -r -a array <<< "$param"
   log_dir=${array[0]}
   length=${array[1]}

   # 执行你的Python脚本并传入log目录和gpuid
   python run.py --log_dir $log_dir --prompt_param 30 ${length} 5
done