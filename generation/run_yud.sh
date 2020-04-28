#!/bin/bash

yudctl run \
        -t process \
        -g 0 -m 50 \
        -p /nfs/users/liuchang/car_comment/mask/mask_comments/  \
        -i registry.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:tf1.12 \
         -- python -m generation.two_compare