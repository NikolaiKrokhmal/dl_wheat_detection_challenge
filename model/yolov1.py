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
        resnet_pre = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet_pre.children())[:-2])

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

        import torch
        import torch.nn as nn
        import torchvision.models as models

        class YOLOv1_ResNet18(nn.Module):
            """
            YOLOv1 with ResNet18 backbone for better performance and transfer learning.
            """

            def __init__(self, grid_size=7, num_boxes=2, num_classes=20, pretrained=True):
                super(YOLOv1_ResNet18, self).__init__()

                self.grid_size = grid_size
                self.num_boxes = num_boxes
                self.num_classes = num_classes
                self.output_size = num_boxes * 5 + num_classes

                # Load pretrained ResNet18 and remove final layers
                resnet = models.resnet18(pretrained=pretrained)

                # Remove avgpool and fc layers (keep only feature extraction)
                self.backbone = nn.Sequential(*list(resnet.children())[:-2])

                # ResNet18 outputs (batch, 512, 14, 14) for 448x448 input
                # We need to adapt this to YOLOv1's requirements

                # Option 1: Add layers to match original YOLOv1 dimensions
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

                # Detection head (same as original YOLOv1)
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

