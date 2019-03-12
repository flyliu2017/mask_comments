#!/usr/bin/env bash

if [ $# -lt 5 ]
then
    echo 'Usage: generate_comments.sh $CONFIG $INPUT_FILE $OUTPUT_DIR $CHECKPOINT_PATH $ITERATION_NUMBER'
    exit 1
fi

pip install opennmt-tf==1.20

i=1
input=$2
while [ $i -le $5 ]
do
    python  infer --config $1 --features_file ${input} --predictions_file $3/pred_${i} --checkpoint_path $4
    exec 3<$input
    exec 4<$3/pred_${i}
    while read line1<&3 && read line2<&4
    do
            echo ${line1/<mask>/ $line2 , <mask>}>>$3/result_${i}
    done
    input=$3/result_${i}
    ((i++))

done