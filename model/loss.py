import torch
import torch.nn as nn
from utils import calculate_iou


class YOLOv1Loss(nn.Module):
    """
    YOLOv1 Loss function exactly as described in the paper.
    """

    def __init__(self, grid_size=7, num_boxes=2, num_classes=1,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()

        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        """
        Calculate YOLOv1 loss.

        Args:
            predictions: (batch_size, S, S, B*5 + C)
            targets: (batch_size, S, S, B*5 + C) - same format as predictions

        Returns:
            Total loss (scalar)
        """
        batch_size = predictions.size(0)

        # Split predictions into components
        # For each grid cell: [x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, class_probs...]
        pred_boxes = predictions[..., :self.num_boxes * 5].contiguous()
        pred_classes = predictions[..., self.num_boxes * 5:]

        # Split targets the same way
        target_boxes = targets[..., :self.num_boxes * 5].contiguous()
        target_classes = targets[..., self.num_boxes * 5:]

        # Reshape boxes: (batch, S, S, B, 5)
        pred_boxes = pred_boxes.view(batch_size, self.grid_size, self.grid_size, self.num_boxes, 5)
        target_boxes = target_boxes.view(batch_size, self.grid_size, self.grid_size, self.num_boxes, 5)

        # Calculate IoU between predicted and target boxes
        iou_b1 = calculate_iou(pred_boxes[..., 0, :], target_boxes[..., 0, :])
        iou_b2 = calculate_iou(pred_boxes[..., 1, :], target_boxes[..., 1, :])

        # Find which box has higher IoU with ground truth (batch, S, S)
        responsible_box = (iou_b1 > iou_b2).float()

        # Object mask: 1 if there's an object in the cell, 0 otherwise (batch, S, S)
        obj_mask = target_boxes[..., 0, 4]  # Use first box confidence as object indicator

        # Choose the responsible box predictions and targets
        # Use broadcasting to select responsible predictions
        pred_responsible = torch.zeros_like(pred_boxes[..., 0, :])  # (batch, S, S, 5)
        target_responsible = target_boxes[..., 0, :]  # Ground truth is in first box slot

        # Select coordinates from responsible box
        for i in range(5):  # x, y, w, h, conf
            pred_responsible[..., i] = (
                    responsible_box * pred_boxes[..., 0, i] +
                    (1 - responsible_box) * pred_boxes[..., 1, i]
            )

        # Coordinate loss (only for cells with objects)
        # Add small epsilon to prevent sqrt of negative numbers
        pred_w = torch.clamp(pred_responsible[..., 2], min=1e-6)
        pred_h = torch.clamp(pred_responsible[..., 3], min=1e-6)
        target_w = torch.clamp(target_responsible[..., 2], min=1e-6)
        target_h = torch.clamp(target_responsible[..., 3], min=1e-6)

        coord_loss = self.lambda_coord * obj_mask * (
                (pred_responsible[..., 0] - target_responsible[..., 0]) ** 2 +  # x
                (pred_responsible[..., 1] - target_responsible[..., 1]) ** 2 +  # y
                (torch.sqrt(pred_w) - torch.sqrt(target_w)) ** 2 +  # w
                (torch.sqrt(pred_h) - torch.sqrt(target_h)) ** 2  # h
        )

        # Confidence loss (object)
        conf_loss_obj = obj_mask * (pred_responsible[..., 4] - target_responsible[..., 4]) ** 2

        # Confidence loss (no object) - penalize both boxes when no object
        conf_loss_noobj = self.lambda_noobj * (1 - obj_mask) * (
                pred_boxes[..., 0, 4] ** 2 + pred_boxes[..., 1, 4] ** 2
        )

        # Class loss (only for cells with objects)
        # Sum over classes, then apply object mask
        class_loss_per_cell = torch.sum((pred_classes - target_classes) ** 2, dim=-1)
        class_loss = obj_mask * class_loss_per_cell

        # Sum all losses
        total_loss = (
                             coord_loss.sum() +
                             conf_loss_obj.sum() +
                             conf_loss_noobj.sum() +
                             class_loss.sum()
                     ) / batch_size

        return total_loss
