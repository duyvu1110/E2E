#!/bin/bash

for layer in 3
do
    for seed in 2
    do
        CUDA_LAUNCH_BLOCKING=1\
        python main_SPN.py  \
        --model_name=Camera_SPN \
        --num_generated=60 \
        --decoder_lr=0.00004 \
        --encoder_lr=0.00002 \
        --data_path=data/smartphone \
        --na_rel_coef=0.2 \
        --num_decoder_layers=$layer \
        --max_epoch=25 \
        --max_grad_norm=10 \
        --random_seed $seed \
        --weight_decay=0.000001 \
        --lr_decay=0.02 \
        --stage=two \
        --multi_heads=5 \
        --method_stage=method_one \

    done
done



# for layer in 3
# do
#     for seed in 2
#     do
#         CUDA_LAUNCH_BLOCKING=1\
#         python main_SPN.py  \
#         --model_name=Camera_SPN_test_Amsgrad \
#         --num_generated=60 \
#         --decoder_lr=0.00004 \
#         --encoder_lr=0.00002 \
#         --data_path=data/smartphone \
#         --na_rel_coef=0.2 \
#         --num_decoder_layers=$layer \
#         --max_epoch=25 \
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

