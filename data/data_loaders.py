import torch
from .wheat_dataset import WheatDataset
from torch.utils.data import random_split, DataLoader
from .transforms import get_val_transforms, get_train_transforms


def get_all_dataloaders(data_dir, csv_file, grid_size, batch_size=16, val_split=0.2, seed=42):

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create full dataset
    full_dataset = WheatDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        apply_mosaic=True,
        grid_size=grid_size,
        transforms=get_train_transforms()
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    # Perform split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply different transforms to validation set
    val_dataset.dataset.transforms = get_val_transforms()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, val_loader
