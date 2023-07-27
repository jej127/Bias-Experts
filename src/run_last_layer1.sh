#!/bin/sh
n=1

#For MNLI
learning_rate=3e-5
warmup_steps=2000
weight_decay=0.1
train_data='MNLI'
custom_teacher='../experiments/bert_weak_nli/eval_mnli_train_answers.json'
lamda=0.3

mode='poe'
custom_teacher='../experiments/bert_weak_nli/eval_mnli_train_answers_0.20.json'

seed=206
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_wm --custom_teacher $custom_teacher --train_data $train_data --do_train --do_eval --mode $mode --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs 3.0 --warmup_steps $warmup_steps --lamda $lamda --adam_epsilon 1e-8 --postfix mode:LastLayer/lamda:$lamda/mode:$mode/q:0.20/seed:$seed

seed=211
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_wm --custom_teacher $custom_teacher --train_data $train_data --do_train --do_eval --mode $mode --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs 3.0 --warmup_steps $warmup_steps --lamda $lamda --adam_epsilon 1e-8 --postfix mode:LastLayer/lamda:$lamda/mode:$mode/q:0.20/seed:$seed

seed=222
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_wm --custom_teacher $custom_teacher --train_data $train_data --do_train --do_eval --mode $mode --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs 3.0 --warmup_steps $warmup_steps --lamda $lamda --adam_epsilon 1e-8 --postfix mode:LastLayer/lamda:$lamda/mode:$mode/q:0.20/seed:$seed

seed=234
CUDA_VISIBLE_DEVICES=$n python train_bert_poe_td.py --output_dir ../experiments/bert_wm --custom_teacher $custom_teacher --train_data $train_data --do_train --do_eval --mode $mode --seed $seed --learning_rate $learning_rate --weight_decay $weight_decay --num_train_epochs 3.0 --warmup_steps $warmup_steps --lamda $lamda --adam_epsilon 1e-8 --postfix mode:LastLayer/lamda:$lamda/mode:$mode/q:0.20/seed:$seed
