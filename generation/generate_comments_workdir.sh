#!/usr/bin/env bash

if [ $# -lt 5 ]
then
    echo 'Usage: generate_comments.sh $WORK_DIR $INPUT_FILE $CHECKPOINT_PATH $ITERATION_NUMBER'
    exit 1
fi

cd $1

i=1
input=$2
out_dir=$1/generation
while [ $i -le $5 ]
do
    python  -m opennmt-tf.infer --config $1/train_5_80.yml --features_file ${input} --predictions_file ${out_dir}/pred_${i} --checkpoint_path $4
    exec 3<$input
    exec 4<${out_dir}/pred_${i}
    while read line1<&3 && read line2<&4
    do
            echo ${line1/<mask>/ $line2 , <mask>}>>$3/result_${i}
    done
    input=${out_dir}/result_${i}
    ((i++))

done