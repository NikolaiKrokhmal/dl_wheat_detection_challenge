import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import decode_outputs


def train(model, train_loader, val_loader, optimizer, criterion, epochs=50, device='cuda'):
    model.to(device)

    history = {
        'train_loss': [],
        'train_mAP': [],
        'val_loss': [],
        'val_mAP': []
    }

    avg_train_loss = []
    avg_val_loss = []
    val_mAP_list = []
    train_map_list = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        metric = MeanAveragePrecision(box_format='xywh', iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])

        preds_list = []
        targets_list = []

        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            ############# convert prediction
            preds_list_batch = []
            targets_list_batch = []

            for i in range(targets.size(0)):
                decoded_pred = decode_outputs(predictions[i])  # pred: (B,7,7,11)
                decoded_target = decode_outputs(targets[i])  # target: (B,7,7,11)

                preds_list.append(decoded_pred)
                preds_list_batch.append(decoded_pred) # To remove

                targets_list.append(decoded_target)
                targets_list_batch.append(decoded_target) # To remove

            metric.update(preds_list_batch, targets_list_batch)
            result = metric.compute()
            print("Train mAP for batch:", result["map"].item())

            # Loss computation
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss.append(train_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss[epoch]:.4f}")

        metric.update(preds_list, targets_list)
        train_mAP = metric.compute()
        train_map_list.append(train_mAP)
        print("Train mAP for epoch:", train_mAP["map"].item())

        # Optional: evaluate on validation set
        val_loss, val_mAP = evaluate(model, val_loader, criterion, device)
        avg_val_loss.append(val_loss)
        val_mAP_list.append(val_mAP)
        print(f"           Validation Loss: {avg_val_loss[epoch]:.4f}")

    # Update history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_mAP'].append(train_map_list)
    history['val_mAP'].append(val_mAP_list)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1,epochs+1), avg_train_loss, 'b-o', label='Training Loss')
    plt.plot(range(1,epochs+1), avg_val_loss, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return history

@torch.no_grad()
def evaluate(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0

    metric = MeanAveragePrecision(box_format='xywh', iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])

    preds_list = []
    targets_list = []

    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model.predict(images)

        ############# convert prediction
        preds_list_batch = []
        targets_list_batch = []

        for i in range(targets.size(0)):
            decoded_pred = decode_outputs(predictions[i])  # pred: (B,7,7,11)
            decoded_target = decode_outputs(targets[i])  # target: (B,7,7,11)
            preds_list.append(decoded_pred)
            preds_list_batch.append(decoded_pred)
            targets_list.append(decoded_target)
            targets_list_batch.append(decoded_target)

        metric.update(preds_list_batch, targets_list_batch)
        result = metric.compute()
        print("Val mAP for batch:", result["map"].item())

        loss = criterion(predictions, targets)
        val_loss += loss.item()

    metric.update(preds_list, targets_list)
    mAP = metric.compute()
    print("Val mAP for epoch:", mAP["map"].item())

    return  val_loss / len(val_loader), mAP
