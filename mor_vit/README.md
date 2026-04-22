MOR-ViT: Mixture of Recursions for Vision Transformer

This repository contains the implementation of MOR-ViT, an adaptive-depth Vision Transformer based on the Mixture of Recursions (MoR) framework. It adapts the dynamic recursion and expert-choice routing from the MoR paper to vision tasks, achieving parameter reduction (~68%) and speedup (~2x) as outlined in the MOR-ViT paper.

Features





Adaptive Depth: Uses expert-choice routing to dynamically determine computation depth for image patches.



Parameter Efficiency: Shares transformer blocks across recursion cycles.



Support for ImageNet-1K: Pretrained on ImageNet-1K with transfer learning capabilities.



Google Colab Compatible: Optimized for single-GPU training on Colab.

Installation on Colab





Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')



Install dependencies: !pip install -r requirements.txt



Upload or clone this repository to Colab: !git clone https://your-repo-url.git mor_vit

Usage





Train: !bash scripts/train_eval.sh -c conf/train/example.yaml -m train



Evaluate on ImageNet: !python evaluate_imagenet.py



Transfer Evaluation: !python eval_transfer.py

Directory Structure





conf/: Configuration files for training and evaluation.



model/mor_model/: Core model implementations.



vision_dataset/: Dataset loading and download scripts.



checkpoints/: Saved model checkpoints.

License

MIT License