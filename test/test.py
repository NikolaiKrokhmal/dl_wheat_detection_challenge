import os
import torch
import cv2
import numpy as np
from utils import extract_bboxes, plot_bboxes


def test(model, test_dir, conf_thresh=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # ImageNet normalization values (RGB format)
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        image = cv2.imread(image_path)

        resized_image = cv2.resize(image, (448, 448))
        resized_image = resized_image.astype(np.float32) / 255.0

        resized_image = (resized_image - imagenet_mean) / imagenet_std

        resized_image = resized_image.transpose(2, 0, 1)

        resized_image = np.expand_dims(resized_image, axis=0)

        tensor_img = torch.from_numpy(resized_image).float()
        tensor_img = tensor_img.to(device)

        with torch.no_grad():
            raw_pred = model.predict(tensor_img)
            boxes = extract_bboxes(raw_pred, conf_threshold=conf_thresh, model_img_size=448, target_img_size=1024)

        plot_bboxes(image, boxes)
