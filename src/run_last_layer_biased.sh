#!/bin/sh
n=1

#For MNLI
learning_rate=3e-5
warmup_steps=2000
train_data='MNLI'
training_dynamics_path='./dynamics/training_dynamics_5_MNLI.json'
weight_decay=0.1
bert_model='google/bert_uncased_L-2_H-128_A-2'
num_train_epochs=3.0

seed=111
q=0.20
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_weak_nli --training_dynamics_path $training_dynamics_path --train_data $train_data --bert_model $bert_model --do_train --do_eval --do_eval_on_train --mode bin_td --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs $num_train_epochs --warmup_steps $warmup_steps --adam_epsilon 1e-8 --q $q

