#!/bin/sh
n=2

#For MNLI
learning_rate=3e-5
warmup_steps=2000
weight_decay=0.1
train_data='MNLI'
custom_teacher='../experiments/bert_weak_nli/eval_mnli_train_answers.json'
lamda=0.3
tau=1.4

#For FEVER
#learning_rate=2e-5
#warmup_steps=1000
#weight_decay=0.1
#train_data='FEVER'
#custom_teacher='../experiments/bert_weak_f/eval_fever_train_answers.json'

#For QQP
#learning_rate=2e-5
#warmup_steps=1000
#weight_decay=0.1
#train_data='QQP'
#custom_teacher='../experiments/bert_weak_f/eval_qqp_train_answers.json'

mode='poe'
custom_teacher='../experiments/bert_weak_nli/eval_mnli_train_answers_0.20.json'

seed=111
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_wm --custom_teacher $custom_teacher --train_data $train_data --do_train --do_eval --mode $mode --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs 3.0 --warmup_steps $warmup_steps --lamda $lamda --adam_epsilon 1e-8 --tau $tau --postfix mode:LastLayer/lamda:$lamda/mode:$mode/q:0.20/seed:$seed 
