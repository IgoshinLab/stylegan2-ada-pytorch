#!/bin/bash
degree=4
for i in `seq 1 10`
do
    sleep 1 & # 提交到后台的任务
    echo $i
    [ `expr $i % $degree` -eq 0 ] && wait
done