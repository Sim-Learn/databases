#!/bin/bash

# 设置 VASP 作业的根目录
ROOT_DIR="/public/home/xuhui/xh/gre/MM/cixing/Co/Co-3-sol"

# 遍历根目录下的所有子文件夹
for SUBDIR in $ROOT_DIR/*; do
  if [ -d "$SUBDIR" ]; then
    echo "Processing directory: $SUBDIR"
    cd "$SUBDIR"
    # 检查是否存在 VASP 输入文件
    if [ -f "INCAR" ] && [ -f "KPOINTS" ] && [ -f "POSCAR" ] && [ -f "POTCAR" ]; then
      echo "Submitting job in $SUBDIR"
      qsub vasp.pbs
    else
      echo "Required files not found in $SUBDIR"
    fi
    cd $ROOT_DIR
  fi
done


