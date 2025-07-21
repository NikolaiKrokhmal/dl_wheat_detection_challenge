import torch
import torch.nn as nn
from torchvision import models


class Yolov1(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.output_size = num_boxes * 5 + num_classes

        # Load pretrained ResNet-18 as backbone (exclude last 2 layers which are not feature extractors)
        resnet_pre = models.resnet18(models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet_pre.children())[:-2])

        # Freeze early layers for domain adaptation
        layers_to_freeze = list(self.backbone.children())[:6]  # First 6 layers
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # adapt ResNet18 output for yolov1 detection head
        self.adapter = nn.Sequential(
            # Increase channels from 512 to 1024
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # Reduce spatial size from 14x14 to 7x7
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # Final conv layers (similar to original YOLOv1)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Detection head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * self.output_size)
        )

    def forward(self, x):
        """Forward pass through ResNet18 backbone + YOLOv1 head."""
        # Extract features with ResNet18 backbone
        features = self.backbone(x)  # (batch, 512, 14, 14)

        # Adapt to YOLOv1 requirements
        features = self.adapter(features)  # (batch, 1024, 7, 7)

        # Get predictions from classifier
        predictions = self.classifier(features)

        # Reshape to grid format
        batch_size = x.size(0)
        predictions = predictions.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.output_size
        )

        return predictions

    def predict(self, x, conf_threshold=0.5, nms_threshold=0.5):
        """Inference method with post-processing"""
        self.eval()
        with torch.no_grad():
            raw_predictions = self.forward(x)
            detections = post_process_predictions(raw_predictions, conf_threshold, nms_threshold)
        return detections


def post_process_predictions(predictions, conf_threshold=0.25, nms_threshold=0.5):
    """
    Post-process YOLOv1 predictions
    Args:
        predictions: (batch_size, S, S, B*5 + C) tensor
        conf_threshold: confidence threshold for filtering
        nms_threshold: IoU threshold for NMS
    """
    batch_size, S, _, _ = predictions.shape

    all_detections = []

    for batch_idx in range(batch_size):
        detections = []

        for i in range(S):
            for j in range(S):
                # Extract predictions for this grid cell
                cell_pred = predictions[batch_idx, i, j]

                # Get both bounding boxes
                box1 = cell_pred[:5]  # [x, y, w, h, conf]
                box2 = cell_pred[5:10]  # [x, y, w, h, conf]
                class_probs = cell_pred[10:]  # class probabilities

                # Process each box if confidence > threshold
                for box_idx, box in enumerate([box1, box2]):
                    conf = box[4].item()
                    if conf > conf_threshold:
                        # Convert to absolute coordinates
                        x_center = (j + box[0]) / S  # Normalize to [0,1]
                        y_center = (i + box[1]) / S
                        width = box[2]
                        height = box[3]

                        # Convert to [x1, y1, x2, y2] format for NMS
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        # Get class with the highest probability
                        class_conf, class_id = torch.max(class_probs, dim=0)
                        final_conf = conf * class_conf.item()

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': final_conf,
                            'class_id': class_id.item()
                        })

        # Apply NMS
        if detections:
            boxes = torch.tensor([d['bbox'] for d in detections])
            scores = torch.tensor([d['confidence'] for d in detections])

            keep_indices = nms(boxes, scores, nms_threshold)
            final_detections = [detections[i] for i in keep_indices]
        else:
            final_detections = []

        all_detections.append(final_detections)

    return all_detections


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        boxes: (N, 4) tensor [x1, y1, x2, y2]
        scores: (N,) tensor of confidence scores
        iou_threshold: IoU threshold for suppression
    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)

    # Sort by confidence scores
    _, indices = scores.sort(descending=True)

    keep = []
    while len(indices) > 0:
        # Keep the box with the highest confidence
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[indices[1:]]

        ious = calculate_iou_nms(current_box, remaining_boxes)

        # Keep boxes with IoU < threshold
        indices = indices[1:][ious < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def calculate_iou_nms(box1, box2):
    """Calculate IoU for NMS (box format: [x1, y1, x2, y2])"""
    # Calculate intersection
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)
