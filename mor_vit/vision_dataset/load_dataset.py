import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(name='cifar10', data_dir='./data', batch_size=32, num_workers=2):
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        val = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
    else:
        raise ValueError(f"Unknown dataset: {name}")
