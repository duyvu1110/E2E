#!/bin/bash

python main_SPN.py  \
--model_name=Camera_SPN \
--num_generated=60 \
--decoder_lr=0.00004 \
--encoder_lr=0.00002 \
--data_path=data/Camera-COQE \
--bert_directory=/home/qtxu/PLM/bert-base-uncased \
--batch_size=4 \
--na_rel_coef=0.2 \
--num_decoder_layers $layer \
--max_epoch=50 \
--max_grad_norm=10 \
--random_seed $seed \
--weight_decay=0.000001 \
--lr_decay=0.02 \
--stage=two \
--multi_heads=5 \
--method_stage=method_one \

# for layer in 3
# do
#     for seed in 123
#     do
#         CUDA_VISIBLE_DEVICES=6 python main_SPN.py  \
#         --model_name=Camera_SPN \
#         --num_generated=60 \
#         --decoder_lr=0.00004 \
#         --encoder_lr=0.00002 \
#         --data_path=data/Camera-COQE \
#         --bert_directory=/home/qtxu/PLM/bert-base-uncased \
#         --batch_size=4 \
#         --na_rel_coef=0.2 \
#         --num_decoder_layers $layer \
#         --max_epoch=50 \
#         --max_grad_norm=10 \
#         --random_seed $seed \
#         --weight_decay=0.000001 \
#         --lr_decay=0.02 \
#         --stage=two \
#         --multi_heads=5 \
#         --method_stage=method_one \

#     done
# done



# for layer in 3
# do
#     for seed in 321
#     do
#         CUDA_VISIBLE_DEVICES=3 python main_SPN.py  \
#         --model_name=Camera_SPN_test_Amsgrad \
#         --num_generated=60 \
#         --decoder_lr=0.00004 \
#         --encoder_lr=0.00002 \
#         --data_path=data/Camera-COQE \
#         --bert_directory=/home/qtxu/PLM/bert-base-uncased \
#         --batch_size=4 \
#         --na_rel_coef=0.2 \
#         --num_decoder_layers $layer \
#         --max_epoch=50 \
#         --max_grad_norm=10 \
#         --random_seed $seed \
#         --weight_decay=0.000001 \
#         --lr_decay=0.02 \
#         --stage=two \
#         --multi_heads=5 \
#         --method_stage=method_one \
#         --data_type=five \

#     done
# done

