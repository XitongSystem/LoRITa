#!/bin/bash

COUNTER=0
gpu_arr=(4 5 6 7)
LEN=${#gpu_arr[@]}
PARALL=4


for dataset in cifar10
do
	for model in vgg16 resnet20
	do

		for factor in 2 3
		do
		   	CUDA_VISIBLE_DEVICES=${gpu_arr[$((COUNTER%LEN))]} python3 -u post_cnns.py --dataset $dataset \
		   	--factor $factor --model $model --folder "base/${dataset}/${model}" -g iter_training --finetune \
		   	--save_folder compress_cnn -cf >> "${model}_${factor}.out" &

		   	COUNTER=$((COUNTER + 1))                    
            if [ $((COUNTER%PARALL)) -eq 0 ]
            then
                wait
            fi

		done

	done
done