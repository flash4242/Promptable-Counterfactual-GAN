import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_dataloaders(batch_size=128, num_workers=4, data_dir=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1] scaling
    ])

    full_train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=False)

    train_indices, valid_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.1,
        stratify=full_train_dataset.targets
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    valid_dataset = Subset(full_train_dataset, valid_indices)

    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, full_train_dataset
