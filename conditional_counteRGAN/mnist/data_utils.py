import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_dataloaders(batch_size=128, num_workers=4, data_dir=None, digits=(2, 5, 8)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1] scaling
    ])

    # Load full dataset
    full_train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=False)

    # Filter only desired digits
    mask = torch.isin(full_train_dataset.targets, torch.tensor(digits))
    full_train_dataset.data = full_train_dataset.data[mask]
    full_train_dataset.targets = full_train_dataset.targets[mask]

    # Remap labels to 0..N-1 so classifier output matches num_classes
    label_map = {digit: idx for idx, digit in enumerate(digits)}
    full_train_dataset.targets = torch.tensor([label_map[int(t)] for t in full_train_dataset.targets])

    # Train/valid split
    train_indices, valid_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.1,
        stratify=full_train_dataset.targets
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    valid_dataset = Subset(full_train_dataset, valid_indices)

    # Test set (apply same filtering & remapping)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=False)
    mask_test = torch.isin(test_dataset.targets, torch.tensor(digits))
    test_dataset.data = test_dataset.data[mask_test]
    test_dataset.targets = test_dataset.targets[mask_test]
    test_dataset.targets = torch.tensor([label_map[int(t)] for t in test_dataset.targets])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, full_train_dataset
