import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import nms

def test(model, dataloader, device, conf_thresh=0.4, iou_thresh=0.5, num_classes=2, max_batches=5):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            images = images.to(device)
            outputs = model(images)  # shape: [B, S, S, B*5 + C]

            for img_idx in range(images.shape[0]):
                output = outputs[img_idx].cpu().numpy()
                S = output.shape[0]
                B = 2  # number of predicted boxes per cell
                C = num_classes

                boxes = []
                scores = []
                labels = []

                for row in range(S):
                    for col in range(S):
                        for b in range(B):
                            conf = output[row,col,b*5+4]
                            if conf > conf_thresh:
                                bx = output[row,col,b*5]
                                by = output[row,col,b*5+1]
                                bw = output[row,col,b*5+2]
                                bh = output[row,col,b*5+3]
                                class_probs = output[row,col,B*5:]
                                class_idx = np.argmax(class_probs)
                                class_conf = class_probs[class_idx] * conf

                                x = (col + bx) / S
                                y = (row + by) / S
                                w = bw
                                h = bh

                                x1 = x - w/2
                                y1 = y - h/2
                                x2 = x + w/2
                                y2 = y + h/2

                                boxes.append([x1, y1, x2, y2])
                                scores.append(class_conf)
                                labels.append(class_idx)

                if len(boxes) == 0:
                    continue

                boxes = torch.tensor(boxes)
                scores = torch.tensor(scores)
                labels = torch.tensor(labels)

                keep = nms(boxes, scores, iou_thresh)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Plotting
                img_np = images[img_idx].permute(1,2,0).cpu().numpy()
                plt.figure(figsize=(6,6))
                plt.imshow(img_np)
                ax = plt.gca()

                for box, score, label in zip(boxes, scores, labels):
                    x1,y1,x2,y2 = box
                    ax.add_patch(plt.Rectangle((x1*img_np.shape[1], y1*img_np.shape[0]),
                                               (x2-x1)*img_np.shape[1],
                                               (y2-y1)*img_np.shape[0],
                                               edgecolor='lime', facecolor='none', linewidth=2))
                    ax.text(x1*img_np.shape[1], y1*img_np.shape[0]-5,
                            f"Class {label}, {score:.2f}",
                            color='lime', fontsize=8, backgroundcolor='black')

                plt.axis('off')
                plt.title("YOLOv1 Predictions (no GT)")
                plt.show()
