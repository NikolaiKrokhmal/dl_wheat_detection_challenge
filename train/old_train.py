import torch
import logging
from tqdm import tqdm
import numpy as np
from utils import set_seed
from models import YOLOv1Loss
from pathlib import Path

def evaluate_epoch_map(model, dataloader, device, confidence_threshold: float = 0.25,
                       dataset_name: str = "dataset") -> float:
    """
    Fast GPU-accelerated mAP evaluation

    Args:
        model: PyTorch models (will be set to eval mode automatically)
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        confidence_threshold: Minimum confidence for predictions
        dataset_name: Name for progress bar

    Returns:
        Mean Average Precision across the entire dataset
    """
    was_training = model.training
    model.eval()

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    all_ap_scores = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Evaluating {dataset_name} mAP')

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Get models predictions
            outputs = model(images)



            processed_output = outputs[0] if isinstance(outputs, tuple) else outputs

            # Fast NMS
            nms_predictions = non_max_suppression_fast(processed_output, conf_thres=confidence_threshold)

            # Convert ground truth
            batch_size = len(nms_predictions)
            ground_truths = targets_to_ground_truth_fast(targets, batch_size)

            # Process each image
            for img_idx in range(batch_size):
                pred = nms_predictions[img_idx]
                gt_list = ground_truths[img_idx]

                # Handle special cases
                if len(gt_list) == 0:
                    if len(pred) > 0:
                        all_ap_scores.append([0.0] * len(iou_thresholds))
                    continue

                if len(pred) == 0:
                    all_ap_scores.append([0.0] * len(iou_thresholds))
                    continue

                # Extract predictions and ground truth
                pred_boxes = pred[:, :4]  # xyxy format
                pred_scores = pred[:, 4]
                gt_boxes = torch.tensor([g['bbox'] for g in gt_list],
                                        device=device, dtype=torch.float32)

                # Fast AP calculation
                ap_scores = calculate_ap_vectorized(pred_boxes, pred_scores, gt_boxes, iou_thresholds)
                all_ap_scores.append(ap_scores)

            # Update progress
            if all_ap_scores:
                current_map = torch.tensor(all_ap_scores).mean().item()
                pbar.set_postfix({'mAP': f'{current_map:.4f}'})

    # Restore training mode
    if was_training:
        model.train()

    # Calculate final mAP
    final_map = torch.tensor(all_ap_scores).mean().item() if all_ap_scores else 0.0
    print(f'{dataset_name.capitalize()} mAP: {final_map:.4f} (evaluated on {len(all_ap_scores)} images)')

    return final_map


def train_single_run(
        seed,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        epochs=50,
        learning_rate=0.001,
        device='cuda',
        save_dir='runs/wheat',
):
    """
    Main training function for wheat detection
    """
    # Setup
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = model.to(device)

    # Loss and optimizer
    criterion = YOLOv1Loss(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'train_mAP': [],
        'val_loss': [],
        'val_mAP': []
    }

    # Best models tracking
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    logging.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision
            outputs = model(images)
            loss, loss_items = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            train_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

        train_map = 0.0
        if (epoch + 1) % 2 == 0:
            print(f"\nEvaluating training mAP...")
            train_map = evaluate_epoch_map(model, train_loader, device, dataset_name="train")

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)

                val_losses.append(loss.item())

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })

        val_map = 0.0
        if (epoch + 1) % 2 == 0:
            print(f"\nEvaluating validation mAP...")
            val_map = evaluate_epoch_map(model, val_loader, device, dataset_name="val")

        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mAP'].append(train_map)
        history['val_mAP'].append(val_map)

        # Learning rate scheduling
        scheduler.step()

        # Logging
        logging.info(
            f'Epoch {epoch + 1}: '
            f'Train Loss: {avg_train_loss:.4f} '
            f'[box: {avg_train_box:.4f}, cls: {avg_train_cls:.4f}, dfl: {avg_train_dfl:.4f}] '
            f'Val Loss: {avg_val_loss:.4f} '
            f'[box: {avg_val_box:.4f}, cls: {avg_val_cls:.4f}, dfl: {avg_val_dfl:.4f}]'
        )

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'history': history
        }

        torch.save(checkpoint, save_dir / 'last.pt')

        # Save best models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_dir / 'best.pt')
            logging.info(f'Best models saved with val loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Save final models and history
    torch.save(history, save_dir / 'history.pt')

    logging.info('Training completed!')
    return model, history


def train_model_3_times(
        model_type,
        train_loader,
        val_loader,
        random_seeds,
        epochs=50,
        learning_rate=0.001,
        device='cuda',
        save_dir='runs/wheat',
):
    list_of_histories = []
    list_of_models = []

    for seed in random_seeds:

        if model_type == 'YOLOv11-s':
            model = YOLOv11(nc=1, ch=3)
        elif model_type == 'MASF-YOLO':
            model = MASF_YOLOv11(nc=1, ch=3)
        else:
            raise ValueError("Options are YOLOv11-s or MASF-YOLO")
        a_model, hist = train_single_run(seed, model, train_loader, val_loader, epochs, learning_rate, device, save_dir)
        list_of_models.append(a_model)
        list_of_histories.append(hist)

    analyzed_results = analyze_multi_run_training(list_of_histories)
    last_model = list_of_models[-1]

    return analyzed_results, last_model