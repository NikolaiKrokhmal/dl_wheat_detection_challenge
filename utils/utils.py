import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import random
import os


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
        # format: [x_min y_min, x_max, y_max]
        x_min, y_min, x_max, y_max, _, _ = bbox
        width = x_max - x_min
        height = y_max - y_min

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


def decode_outputs(output, S=7, B=2, C=1, img_size=448, conf_thresh=0.25):
    boxes = []
    scores = []
    labels = []

    cell_size = img_size / S  # e.g., 448/7 = 64
    for row in range(S):
        for col in range(S):
            for b in range(B):
                # Extract box info
                bx = output[row, col, b*5]
                by = output[row, col, b*5+1]
                bw = output[row, col, b*5+2]
                bh = output[row, col, b*5+3]
                conf = output[row, col, b*5+4]

                # CLASS SCORE: last element in the 11-dim vector
                class_score = output[row, col, B * 5]  # index 10

                # Final confidence = confidence * class probability
                final_score = conf * class_score

                # Skip low-confidence boxes
                if final_score < conf_thresh:
                    continue

                # Convert to image coordinates
                x_center = (col + bx) * cell_size
                y_center = (row + by) * cell_size
                w = bw * img_size
                h = bh * img_size

                # x1 = x_center - w/2
                # y1 = y_center - h/2
                # x2 = x_center + w/2
                # y2 = y_center + h/2

                boxes.append([x_center, y_center, w, h])
                scores.append(conf.item())
                labels.append(1)  # Only 1 class

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


def decode_target(target, S=7, img_size=448):
    boxes = []
    labels = []

    cell_size = img_size / S

    for row in range(S):
        for col in range(S):
            obj = target[row, col, 4]  # confidence indicator
            if obj == 0:
                continue

            bx = target[row, col, 0]
            by = target[row, col, 1]
            bw = target[row, col, 2]
            bh = target[row, col, 3]

            x_center = (col + bx) * cell_size
            y_center = (row + by) * cell_size
            w = bw * img_size
            h = bh * img_size

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # Only 1 class

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


def extract_bboxes(predictions, conf_threshold=0.25, model_img_size=448, target_img_size=1024):
    """
    Extract bounding boxes from raw YOLOv1 predictions and scale to target image size
    """
    batch_size, S, _, _ = predictions.shape

    # Calculate scaling factor
    scale_factor = target_img_size / model_img_size  # 1024 / 448 = 2.286

    boxes = []
    for batch_idx in range(batch_size):
        boxes = []

        for i in range(S):
            for j in range(S):
                cell_pred = predictions[batch_idx, i, j]

                # Extract both boxes from this cell
                box1 = cell_pred[:5]  # [x, y, w, h, conf]
                box2 = cell_pred[5:10]  # [x, y, w, h, conf]
                class_probs = cell_pred[10:]

                class_conf, class_id = torch.max(class_probs, dim=0)

                # Pick the box with higher confidence
                conf1 = box1[4].item()
                conf2 = box2[4].item()

                if conf1 > conf2:
                    best_box = box1
                    best_conf = conf1
                else:
                    best_box = box2
                    best_conf = conf2

                # Only process if the best box passes threshold
                if best_conf > conf_threshold:
                    # Convert coordinates
                    x_center = (j + best_box[0]) * model_img_size / S
                    y_center = (i + best_box[1]) * model_img_size / S
                    width = best_box[2].item() * model_img_size
                    height = best_box[3].item() * model_img_size

                    # Convert to corner format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    # Scale to target image size
                    x1 *= scale_factor
                    y1 *= scale_factor
                    x2 *= scale_factor
                    y2 *= scale_factor

                    final_conf = best_conf * class_conf.item()

                    boxes.append([
                        x1.item(), y1.item(), x2.item(), y2.item(),
                        final_conf,
                        class_id.item()
                    ])

    return boxes


if __name__ == '__main__':
    pass
