import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import random
import os


def calculate_dataset_stats(dataloader):
    """Calculate mean and std for your specific dataset"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std


def plot_bboxes(image, bboxes, title="Image with Bboxes"):
    """
    Simple function to plot bounding boxes on an image with error handling

    Args:
        image: numpy array (H, W, C)
        bboxes: list of bboxes in [x_min, y_min, x_max, y_max] format
        title: plot title
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(title)

    for bbox in bboxes:
        # YOLO format: [class, x_center_norm, y_center_norm, width_norm, height_norm]
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = bbox

        # Convert normalized coordinates back to pixel coordinates
        img_height, img_width = image.shape[:2]  # Get actual image dimensions

        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height

        # Convert to top-left corner coordinates for Rectangle
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def set_seed(seed):
    """
    Set seeds for reproducible training results
    """
    # Python random
    random.seed(seed)

    # Numpy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # PyTorch CUDA random (for GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch deterministic (slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Custom collate function for YOLO format"""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, 0)

    # Combine all targets with batch indices
    batch_targets = []
    for i, target in enumerate(targets):
        if len(target) > 0:
            # Add batch index as first column
            batch_idx = torch.full((len(target), 1), i, dtype=target.dtype)
            target_with_batch = torch.cat([batch_idx, target], dim=1)
            batch_targets.append(target_with_batch)

    if batch_targets:
        targets = torch.cat(batch_targets, 0)  # [total_detections, 6]
    else:
        targets = torch.empty(0, 6)

    return images, targets


def calculate_iou(box1, box2):
    """Calculate IoU between two sets of boxes."""
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Calculate intersection
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


if __name__ == '__main__':
    pass
