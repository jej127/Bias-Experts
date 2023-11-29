#!/bin/sh
n=3

#For MNLI
train_data='MNLI'
bert_model='google/bert_uncased_L-2_H-128_A-2'
learning_rate=3e-5
weight_decay=0.1
num_train_epochs=5.0
warmup_steps=2000
logging_steps=2000
adam_epsilon_b=1e-8

seed=222
CUDA_VISIBLE_DEVICES=$n python train_bert_dynamics.py --output_dir ../experiments/bert_weak_nli --bert_model $bert_model --train_data $train_data --do_train --do_eval --mode vn --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs $num_train_epochs --warmup_steps $warmup_steps --logging_steps $logging_steps --adam_epsilon 1e-8
