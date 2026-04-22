#!/bin/bash

# Script to train and evaluate MOR-ViT and ViT models on Colab

# Default configuration
CONFIG="/content/mor_vit/conf/train/example.yaml"
MODE="train"

# Parse arguments
while getopts "c:m:" opt; do
    case $opt in
        c) CONFIG="$OPTARG";;
        m) MODE="$OPTARG";;
        ?) echo "Usage: $0 [-c config_file] [-m mode]"; exit 1;;
    esac
done

case $MODE in
    "train")
        echo "Starting training with config: $CONFIG"
        !python /content/mor_vit/train.py $CONFIG
        ;;
    "eval")
        echo "Starting evaluation with config: $CONFIG"
        !python /content/mor_vit/evaluate_imagenet.py $CONFIG
        ;;
    "transfer")
        echo "Starting transfer evaluation with config: /content/mor_vit/conf/eval_transfer/example.yaml"
        !python /content/mor_vit/eval_transfer.py
        ;;
    *)
        echo "Invalid mode. Use 'train', 'eval', or 'transfer'"
        exit 1
        ;;
esac