#!/usr/bin/env bash

declare -a list=("ori_han" "sent_ori_han" "muil_han" "sent_muil_han" "muil_stock_han" "sent_muil_stock_han")
for model_type in "${list[@]}"
do
echo "$model_type"
python3 train.py --model_type "$model_type"
done

