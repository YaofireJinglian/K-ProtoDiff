#!/bin/bash

names=("EEG" "Stocks" "Exchange" "ETTh" "Electricity" "Energy" "Traffic" "Weather" "Illness" )
for name in "${names[@]}"; do
    python run.py --name $name --config_file ./Config/$name.yaml --gpu 3 --train
    python run.py --name $name --config_file ./Config/$name.yaml --gpu 3 --sample 0 --milestone 10
done