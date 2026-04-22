import os
import torch
from google.colab import drive
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model.mor_model import mor_vit_from_pretrained

def eval_transfer(cfg: DictConfig):
    drive.mount('/content/drive')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mor_vit_from_pretrained(cfg.model_name_or_path, cfg).to(device)
    model.load_state_dict(torch.load(cfg.weights))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root="/content/drive/MyDrive/data/cifar10", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.evaluation.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs["logits"], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Transfer Accuracy on CIFAR-10: {accuracy:.2f}%")

if __name__ == "__main__":
    cfg = OmegaConf.load("/content/mor_vit/conf/eval_transfer/example.yaml")
    eval_transfer(cfg)