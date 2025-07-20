import albumentations as A


def get_train_transforms(mean=(0.2140, 0.3170, 0.3142), std=(0.1747, 0.2089, 0.2061), img_size=448):
    """Training augmentations using Albumentations (your original)"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean, std),
        A.ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        min_visibility=0.1
    ))


def get_val_transforms(mean=(0.2140, 0.3170, 0.3142), std=(0.1747, 0.2089, 0.2061), img_size=448):
    """Validation transforms (no augmentation) (your original)"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean, std),
        A.ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',
        min_visibility=0.1
    ))