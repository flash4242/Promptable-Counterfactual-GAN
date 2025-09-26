import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_dataloaders(batch_size=128, num_workers=4, data_dir=None,
                    cfg=None):
    # clean transform (what GAN sees by default)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1]
    ])

    # create two dataset instances (clean + aug) but share the SAME filtered samples (same split indices)
    full_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=False)

    # create train/valid split indices once
    all_indices = list(range(len(full_dataset)))
    train_indices, valid_indices = train_test_split(all_indices, test_size=0.1, stratify=full_dataset.targets.numpy())

    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)

    # test dataset (clean only)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Return augmented loader first (for classifier training), and clean loader for GAN loop
    return train_loader, valid_loader, test_loader, full_dataset
