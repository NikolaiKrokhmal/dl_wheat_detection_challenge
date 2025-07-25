import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import decode_outputs, set_seed, analyze_multi_run_training
from models import YOLOv1Loss
from pathlib import Path
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import Yolov1, FPNYolo

def train(seed, model, train_loader, val_loader, learning_rate, grid_size, epochs=50, device='cuda'):
    model.to(device)

    set_seed(seed)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = YOLOv1Loss(grid_size, 2, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        metric = MeanAveragePrecision(box_format='xywh', iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])

        preds_list = []
        targets_list = []

        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Loss computation
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_avg_loss = train_loss / len(train_loader)
        avg_train_loss.append(epoch_avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_avg_loss:.4f}")

        if epoch % 5 == 0:
            for i in range(targets.size(0)):
                decoded_pred = decode_outputs(predictions[i])  # pred: (B,S,S,11)
                decoded_target = decode_outputs(targets[i])  # target: (B,S,S,11)

                preds_list.append(decoded_pred)

                targets_list.append(decoded_target)
            metric.update(preds_list, targets_list)
            train_mAP = metric.compute()
            train_mAP = train_mAP['map'].item()
            train_map_list.append(train_mAP)
            print("Train mAP for epoch:", train_mAP)

            # Optional: evaluate on validation set
            val_loss, val_mAP = evaluate(model, val_loader, criterion, device)
            avg_val_loss.append(val_loss)
            val_mAP_list.append(val_mAP)
            print(f"           Validation Loss: {avg_val_loss[epoch // 5]:.4f}")

    # Update history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_mAP'].append(train_map_list)
    history['val_mAP'].append(val_mAP_list)

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, epochs + 1), avg_train_loss, 'b-o', label='Training Loss')
    # plt.plot(range(1, epochs + 1), avg_val_loss, 'r-o', label='Validation Loss')
    # plt.title('Training and Validation Loss Curves')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return model, history


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
        predictions.to(device)

        # convert prediction

        for i in range(targets.size(0)):
            decoded_pred = decode_outputs(predictions[i])  # pred: (B,s,s,11)
            decoded_target = decode_outputs(targets[i])  # target: (B,s,s,11)
            preds_list.append(decoded_pred)
            targets_list.append(decoded_target)

        loss = criterion(predictions, targets)
        val_loss += loss.item()

    metric.update(preds_list, targets_list)
    mAP = metric.compute()
    mAP = mAP['map'].item()
    print("Val mAP for epoch:", mAP)

    return val_loss / len(val_loader), mAP


def train_model_3_times(
        model_type,
        train_loader,
        val_loader,
        random_seeds,
        grid_size,
        epochs=30,
        learning_rate=0.001,
        device='cuda',
        save_dir='runs/',
):
    list_of_histories = []
    list_of_models = []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for seed in random_seeds:
        if model_type == 'yolov1':
            model = Yolov1(grid_size=7, num_boxes=2, num_classes=1)
        elif model_type == 'fpn':
            model = FPNYolo(grid_size=grid_size, num_boxes=2, num_classes=1)
        a_model, hist = train(seed=seed,
                              model=model,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              grid_size=grid_size,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              device=device)
        list_of_models.append(a_model)
        list_of_histories.append(hist)

        weights_path = f'runs/{model_type}_s_{grid_size}__seed_{seed}.pt'
        torch.save(model.state_dict(), weights_path)

    pickle_path = f'runs/{model_type}_s_{grid_size}_list_of_hist.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(list_of_histories, f)

    last_model = list_of_models[-1]

    analyzed_results = analyze_multi_run_training(list_of_histories, model_type)

    return last_model
