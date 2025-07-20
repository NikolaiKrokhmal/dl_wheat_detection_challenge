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

        # Load pretrained ResNet-18 as backbone
        resnet_pre = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet_pre.children())[:-2])

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
