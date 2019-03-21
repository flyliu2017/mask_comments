#!/usr/bin/env bash

python3 -m data_process.processor mask  \
                                            --df_path /data/share/liuchang/car_comment/mask/df_5_80_p5_p10.json    \
                                            --output_dir /data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask    \
                                            --min_words 10    \
                                            --max_words 80    \
                                            --min_phrase 5    \
                                            --max_phrase 10    \
                                            --suffix only_mask    \
                                            --mode random    \
                                            --add_keywords only_mask  &&  \
python3  /data/share/liuchang/car_comment/mask/mask_comments/modification/keywords.py  phrase --suffix only_mask &&  \
python3  /data/share/liuchang/car_comment/mask/mask_comments/modification/change_mask.py
