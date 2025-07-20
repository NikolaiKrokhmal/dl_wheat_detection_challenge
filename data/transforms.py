import albumentations as A


def get_train_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), img_size=448):
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
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1
    ))


def get_val_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), img_size=448):
    """Validation transforms (no augmentation) (your original)"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean, std),
        A.ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1
    ))