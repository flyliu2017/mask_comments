#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 t2t-decoder   \
                        --data_dir /nfs/users/liuchang/comments_dayu/tag_prediction/data   \
                        --t2t_usr_dir /nfs/users/liuchang/comments_dayu/subword   \
                        --tmp_dir=/nfs/users/liuchang/comments_dayu/subword/temp   \
                        --problem comments__prefixed__tag   \
                        --model transformer   \
                        --hparams_set=transformer_base   \
                        --decode_to_file /nfs/users/liuchang/comments_dayu/tag_prediction/t2t_prefix_tag/pred_t2t_tag   \
                        --output_dir /nfs/users/liuchang/comments_dayu/tag_prediction/t2t_prefix_tag/   \
                        --eval_use_test_set \
                        --eval_throttle_seconds=1800

yudctl run -t t2t-train-prefix-tag -p /nfs/users/liuchang/tensor2tensor  \
    -g 1 -i registry.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:tf1.12 -r requirements.txt  \
-- python tensor2tensor/bin/t2t-trainer   \
--data_dir /nfs/users/liuchang/comments_dayu/tag_prediction/data  \
--t2t_usr_dir /nfs/users/liuchang/comments_dayu/subword  \
--tmp_dir=/nfs/users/liuchang/comments_dayu/subword/temp  \
--problem comments__prefixed__tag  \
--model transformer  \
--hparams_set=transformer_base   \
--output_dir /nfs/users/liuchang/comments_dayu/tag_prediction/t2t_prefix_tag/  \
--eval_use_test_set  \
--eval_throttle_seconds=1800  \
--train_steps 500000

t2t-trainer
-data_dir /nfs/users/liuchang/comments_dayu/tag_prediction/data
--t2t_usr_dir /nfs/users/liuchang/comments_dayu/subword
--tmp_dir=/nfs/users/liuchang/comments_dayu/subword/temp
--problem extract__phrases
--model transformer
--hparams_set=transformer_base
--output_dir /nfs/users/liuchang/comments_dayu/tag_prediction/t2t/test
--generate_data



classifier.py \
  --train_records=queries.tfrecords \
  --eval_records=queries.tfrecords \
  --label_file=labels.txt \
  --vocab_file=vocab.txt \
  --model_dir=model \
  --export_dir=model \
   --embedding_dimension=512 \
   --ngram_embedding_dimension=512 \
    --use_ngrams=True \
    --train_steps=100000

python run_classify.py
--task_name=seg-phrase
--do_predict=true
--do_eval=true
--data_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/data
--vocab_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt
--bert_config_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json
--init_checkpoint=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt
--max_seq_length=128
--train_batch_size=32
--learning_rate=2e-5
--num_train_epochs=20.0
--output_dir
/nfs/users/liuchang/comments_dayu/tag_prediction/segtag
--save_checkpoints_steps=100