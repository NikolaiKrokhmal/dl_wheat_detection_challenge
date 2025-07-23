import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import nms


class Yolov1(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.output_size = num_boxes * 5 + num_classes

        # Load pretrained ResNet-18 as backbone (exclude last 2 layers which are not feature extractors)
        resnet_pre = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
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


def post_process_predictions(predictions, conf_threshold=0.5, nms_threshold=0.5):
    """
    Post-process YOLOv1 predictions using torchvision NMS
    """
    batch_size, S, S, _ = predictions.shape
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

                        # Get class with highest probability
                        class_conf, class_id = torch.max(class_probs, dim=0)
                        final_conf = conf * class_conf.item()

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': final_conf,
                            'class_id': class_id.item()
                        })

        # Apply torchvision NMS 🎯
        if detections:
            boxes = torch.tensor([d['bbox'] for d in detections])
            scores = torch.tensor([d['confidence'] for d in detections])

            # Use torchvision NMS - much simpler!
            keep_indices = nms(boxes, scores, nms_threshold)

            final_detections = [detections[i] for i in keep_indices]
        else:
            final_detections = []

        all_detections.append(final_detections)

    return detections_to_tensor(all_detections, batch_size)


def apply_nms(boxes, scores, iou_threshold=0.5):
    """Simple wrapper around torchvision NMS"""
    return nms(boxes, scores, iou_threshold)


def detections_to_tensor(all_detections, batch_size, grid_size=7, num_classes=1):
    """
    Convert post-processed detections back to YOLOv1 tensor format

    Args:
        all_detections: List of length batch_size, each containing detection dicts
        batch_size: Number of images in batch
        grid_size: Grid size (default 7)
        num_classes: Number of classes (default 1)

    Returns:
        tensor: Shape [batch_size, grid_size, grid_size, 2*5 + num_classes]
    """
    # Initialize output tensor
    output_tensor = torch.zeros(batch_size, grid_size, grid_size, 2 * 5 + num_classes)

    for batch_idx, detections in enumerate(all_detections):
        for detection in detections:
            # Extract detection info
            bbox = detection['bbox']  # [x1, y1, x2, y2] in [0,1] coordinates
            confidence = detection['confidence']
            class_id = detection['class_id']

            # Convert [x1, y1, x2, y2] to [x_center, y_center, width, height]
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2  # Center x in [0,1]
            y_center = (y1 + y2) / 2  # Center y in [0,1]
            width = x2 - x1  # Width in [0,1]
            height = y2 - y1  # Height in [0,1]

            # Determine which grid cell this detection belongs to
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)

            # Clamp to valid grid indices
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))

            # Convert to YOLOv1 format (relative to grid cell for x,y)
            cell_x = (x_center * grid_size) - grid_x  # Offset within cell [0,1]
            cell_y = (y_center * grid_size) - grid_y  # Offset within cell [0,1]
            # Width and height remain relative to full image [0,1]

            # Find an available box slot in this grid cell
            # Check first box slot
            if output_tensor[batch_idx, grid_y, grid_x, 4] == 0:  # First box available
                output_tensor[batch_idx, grid_y, grid_x, 0] = cell_x
                output_tensor[batch_idx, grid_y, grid_x, 1] = cell_y
                output_tensor[batch_idx, grid_y, grid_x, 2] = width
                output_tensor[batch_idx, grid_y, grid_x, 3] = height
                output_tensor[batch_idx, grid_y, grid_x, 4] = confidence
            # Check second box slot
            elif output_tensor[batch_idx, grid_y, grid_x, 9] == 0:  # Second box available
                output_tensor[batch_idx, grid_y, grid_x, 5] = cell_x
                output_tensor[batch_idx, grid_y, grid_x, 6] = cell_y
                output_tensor[batch_idx, grid_y, grid_x, 7] = width
                output_tensor[batch_idx, grid_y, grid_x, 8] = height
                output_tensor[batch_idx, grid_y, grid_x, 9] = confidence
            # If both slots are occupied, skip (this shouldn't happen after NMS)

            # Set class probability (for single class, just set to 1.0)
            if num_classes == 1:
                output_tensor[batch_idx, grid_y, grid_x, 10] = 1.0
            else:
                # For multi-class, set the specific class
                output_tensor[batch_idx, grid_y, grid_x, 10 + class_id] = 1.0

    return output_tensor
