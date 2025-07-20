from pathlib import Path
import cv2
import numpy as np
import torch
import random
import pandas as pd
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    """
    Custom Dataset for Global Wheat Detection Challenge

    Expected data structure:
    /data/
        train/
            *.jpg (images)
        train.csv (annotations with columns: image_id, width, height, bbox)

    bbox format in CSV: [x_min, y_min, width, height] (Pascal VOC format)
    """

    def __init__(self, data_dir, csv_file, apply_mosaic, transforms=None, img_size=448):
        """
        Args:
            data_dir (str): Path to the data directory containing images
            csv_file (str): Path to CSV file with annotations
            transforms (albumentations.Compose): Augmentation pipeline
            img_size (int): Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transforms = transforms
        self.apply_mosaic = apply_mosaic

        # Load annotations
        self.df = pd.read_csv(csv_file)

        # Parse bounding boxes from string format if needed
        if isinstance(self.df['bbox'].iloc[0], str):
            self.df['bbox'] = self.df['bbox'].apply(eval)

        # Group by image_id to handle multiple boxes per image
        self.image_ids = self.df['image_id'].unique()

        print(f"Dataset initialized with {len(self.image_ids)} images")
        print(f"Total annotations: {len(self.df)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Processed image tensor [3, H, W]
            target (dict): Contains 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        image_id = self.image_ids[idx]

        # Load image
        img_path = self.data_dir / f"{image_id}.jpg"
        image = cv2.imread(str(img_path))

        # Get all bounding boxes for this image
        img_annotations = self.df[self.df['image_id'] == image_id]

        boxes = []

        for _, row in img_annotations.iterrows():
            bbox = row['bbox']  # [x_min, y_min, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, w, h, 1])

        # Convert to numpy arrays for albumentations
        boxes = np.array(boxes, dtype=np.float32)

        if self.apply_mosaic: #TODO: modify mosaic augmentation for new coordinates
            image, boxes = self.mosaic_augmentation(image, boxes, p=1)

        # Apply transforms if specified
        if self.transforms:
            sample = self.transforms(
                image=image,
                bboxes=boxes,
            )
            image = sample['image']
            boxes = np.array(sample['bboxes'], dtype=np.float32) # [x_min, y_min. w, h]

        grid_tensor = torch.zeros(7, 7, 11)

        # Normalize bboxes for YOLO
        for box in boxes:
            x, y, w, h, c = box

            x_center = (x + w / 2)  #  x_center
            y_center = (y + h / 2)  #  y_center

            cell_size = self.img_size / 7

            grid_x_index = int(x_center / cell_size)
            grid_y_index = int(y_center / cell_size)

            delta_x = (x_center % cell_size) / cell_size
            delta_y = (y_center % cell_size) / cell_size
            delta_h = h / self.img_size
            delta_w = w / self.img_size

            if not grid_tensor[grid_y_index, grid_x_index, 0]:
                grid_tensor[grid_y_index, grid_x_index, 0:5] = torch.tensor([delta_x, delta_y, delta_h, delta_w, c])
            elif not grid_tensor[grid_y_index, grid_x_index, 5]:
                grid_tensor[grid_y_index, grid_x_index, 5:10] = torch.tensor([delta_x, delta_y, delta_h, delta_w, c])

        return image, grid_tensor

    def get_raw_item(self, idx):
        """
        Get raw image and target without any transforms.
        This is needed for mosaic augmentation.
        """
        image_id = self.image_ids[idx]

        # Load raw image
        img_path = self.data_dir / f"{image_id}.jpg"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        img_annotations = self.df[self.df['image_id'] == image_id]

        boxes = []
        labels = []

        for _, row in img_annotations.iterrows():
            bbox = row['bbox']  # [x_min, y_min, width, height]

            # Parse bbox if it's a string
            if isinstance(bbox, str):
                bbox = eval(bbox)

            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            # Ensure coordinates are within image bounds
            img_height, img_width = image.shape[:2]
            x_min = max(0, min(x_min, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            x_max = max(x_min + 1, min(x_max, img_width))
            y_max = max(y_min + 1, min(y_max, img_height))

            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Wheat class

        # Return raw image and bboxes (no COCO format needed for mosaic)
        return image, {
            'bboxes': boxes,
            'labels': labels
        }

    def mosaic_augmentation(self, image, boxes, p=1):
        """
        Apply mosaic augmentation by combining 4 images in a 2x2 grid

        Args:
            image: Current image (H, W, 3)
            boxes: Current bounding boxes in [x_min, y_min, x_max, y_max] format
            labels: Current labels array
            p: Probability of applying mosaic

        Returns:
            mosaic_image: Augmented image (640, 640, 3)
            mosaic_boxes: Adjusted bounding boxes
            mosaic_labels: Adjusted labels (matching the boxes)
        """
        # Check if we should apply mosaic
        if random.random() >= p:
            return image, boxes, labels

        # Get 3 additional random samples
        additional_indices = random.choices(range(len(self)), k=3)

        # Collect all images, bboxes, and labels
        all_images = [image]
        all_bboxes = [boxes.tolist() if isinstance(boxes, np.ndarray) else boxes]
        all_labels = [labels.tolist() if isinstance(labels, np.ndarray) else labels]

        for idx in additional_indices:
            add_img, add_target = self.get_raw_item(idx)
            all_images.append(add_img)
            all_bboxes.append(add_target['bboxes'])
            all_labels.append(add_target['labels'])

        # Create mosaic
        return self._create_mosaic(all_images, all_bboxes, all_labels)

    def _create_mosaic(self, images, all_bboxes, all_labels):
        """
        Create a 2x2 mosaic from 4 images and randomly crop 640x640

        Args:
            images: List of 4 images, each (1024, 1024, 3)
            all_bboxes: List of 4 bbox lists
            all_labels: List of 4 label lists

        Returns:
            cropped_mosaic: (640, 640, 3) image
            final_bboxes: Adjusted bounding boxes
            final_labels: Labels corresponding to final_bboxes
        """
        # Step 1: Shuffle the images, bboxes, and labels together for randomness
        combined = list(zip(images[:4], all_bboxes[:4], all_labels[:4]))
        random.shuffle(combined)
        shuffled_images, shuffled_bboxes, shuffled_labels = zip(*combined)

        # Step 2: Create 2x2 grid (2048x2048 total)
        # Concatenate top row (img[0] + img[1])
        top_row = np.concatenate([shuffled_images[0], shuffled_images[1]], axis=1)

        # Concatenate bottom row (img[2] + img[3])
        bottom_row = np.concatenate([shuffled_images[2], shuffled_images[3]], axis=1)

        # Concatenate top and bottom rows
        large_mosaic = np.concatenate([top_row, bottom_row], axis=0)  # Shape: (2048, 2048, 3)

        # Step 3: Adjust bounding boxes to large mosaic coordinates
        adjusted_bboxes = []
        adjusted_labels = []

        # Offsets for each quadrant in the 2048x2048 mosaic
        offsets = [
            (0, 0),  # Top-left: img[0]
            (1024, 0),  # Top-right: img[1]
            (0, 1024),  # Bottom-left: img[2]
            (1024, 1024)  # Bottom-right: img[3]
        ]

        for i, (bboxes, labels, (offset_x, offset_y)) in enumerate(zip(shuffled_bboxes, shuffled_labels, offsets)):
            for bbox, label in zip(bboxes, labels):
                if len(bbox) >= 4:  # Ensure valid bbox format
                    x_min, y_min, x_max, y_max = bbox[:4]

                    # Translate bbox to large mosaic coordinates
                    new_x_min = x_min + offset_x
                    new_y_min = y_min + offset_y
                    new_x_max = x_max + offset_x
                    new_y_max = y_max + offset_y

                    adjusted_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
                    adjusted_labels.append(label)  # Keep corresponding label

        # Step 4: Randomly crop 640x640 from the 2048x2048 mosaic
        max_crop_x = 2048 - 640  # Fixed: should be 640, not 940
        max_crop_y = 2048 - 640

        crop_x = random.randint(0, max_crop_x)
        crop_y = random.randint(0, max_crop_y)

        # Crop the image
        cropped_mosaic = large_mosaic[crop_y:crop_y + 640, crop_x:crop_x + 640]

        # Step 5: Adjust bounding boxes for the crop and filter
        final_bboxes = []
        final_labels = []

        for bbox, label in zip(adjusted_bboxes, adjusted_labels):
            x_min, y_min, x_max, y_max = bbox

            # Translate bbox coordinates relative to crop
            new_x_min = x_min - crop_x
            new_y_min = y_min - crop_y
            new_x_max = x_max - crop_x
            new_y_max = y_max - crop_y

            # Clip to crop boundaries (0, 0, 640, 640)
            clipped_x_min = max(0, new_x_min)
            clipped_y_min = max(0, new_y_min)
            clipped_x_max = min(640, new_x_max)
            clipped_y_max = min(640, new_y_max)

            # Check if bbox is still valid after clipping
            if (clipped_x_max > clipped_x_min and
                    clipped_y_max > clipped_y_min and
                    (clipped_x_max - clipped_x_min) >= 8 and  # Minimum width
                    (clipped_y_max - clipped_y_min) >= 8):  # Minimum height

                final_bboxes.append([clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max])
                final_labels.append(label)  # Keep corresponding label

        return cropped_mosaic, final_bboxes, final_labels