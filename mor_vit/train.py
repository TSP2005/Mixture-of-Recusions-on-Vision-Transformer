import os
import torch
from google.colab import drive
from omegaconf import DictConfig, OmegaConf
from model.mor_model import mor_vit_from_pretrained, vit_from_pretrained
from vision_dataset.load_dataset import load_dataset

def train(cfg: DictConfig):
    drive.mount('/content/drive')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model == 'mor_vit':
        model = mor_vit_from_pretrained(cfg.model_name_or_path, cfg).to(device)
    else:
        model = vit_from_pretrained(cfg.model_name_or_path).to(device)

    train_loader = load_dataset(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    for step in range(cfg.num_train_steps):
        for batch in train_loader:
            pixel_values = batch[0].to(device)
            attention_mask = None  # ViT doesn't use this natively
            outputs = model(pixel_values)
            loss = outputs["logits"].mean() + (outputs.get("router_z_loss", 0) if cfg.model == 'mor_vit' else 0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % cfg.save_interval == 0:
            torch.save(model.state_dict(), f"{cfg.output_dir}/checkpoint_{step}.pt")

if __name__ == "__main__":
    cfg_path = "/content/mor_vit/conf/train/example.yaml" if len(os.sys.argv) < 2 else os.sys.argv[1]
    cfg = OmegaConf.load(cfg_path)
    train(cfg)