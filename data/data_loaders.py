import torch
from .wheat_dataset import WheatDataset
from torch.utils.data import random_split, DataLoader
from .transforms import get_val_transforms, get_train_transforms
from utils import collate_fn


def get_all_dataloaders(data_dir, csv_file, batch_size=16, val_split=0.2, seed=42):

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create full dataset
    full_dataset = WheatDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        apply_mosaic=False,
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

    aug_full_dataset = WheatDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        apply_mosaic=True,
        transforms=get_train_transforms()
    )

    # Perform split for augmented
    aug_dataset, _ = random_split(
        aug_full_dataset,
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
        collate_fn=collate_fn,
        drop_last=True
    )

    aug_loader = DataLoader(
        aug_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    return train_loader, aug_loader, val_loader
