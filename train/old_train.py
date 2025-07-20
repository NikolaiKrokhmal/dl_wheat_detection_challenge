import torch
import logging
from tqdm import tqdm
import numpy as np
from utils import set_seed
from model import YOLOv1Loss
from pathlib import Path


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

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    history = {
        'train_loss': [],
        'train_bbox_loss': [],
        'train_cls_loss': [],
        'train_dfl_loss': [],
        'train_mAP': [],
        'val_loss': [],
        'val_bbox_loss': [],
        'val_cls_loss': [],
        'val_dfl_loss': [],
        'val_mAP': []
    }

    # Best model tracking
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    logging.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_box_losses = []
        train_cls_losses = []
        train_dfl_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Record losses
            train_losses.append(loss.item())
            train_box_losses.append(loss_items[0].item())
            train_cls_losses.append(loss_items[1].item())
            train_dfl_losses.append(loss_items[2].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items[0]:.4f}',
                'cls': f'{loss_items[1]:.4f}',
                'dfl': f'{loss_items[2]:.4f}'
            })

        train_map = 0.0
        if (epoch + 1) % 2 == 0:
            print(f"\nEvaluating training mAP...")
            train_map = evaluate_epoch_map(model, train_loader, device, dataset_name="train")

        # Validation phase
        model.eval()
        val_losses = []
        val_box_losses = []
        val_cls_losses = []
        val_dfl_losses = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_box_losses.append(loss_items[0].item())
                val_cls_losses.append(loss_items[1].item())
                val_dfl_losses.append(loss_items[2].item())

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'box': f'{loss_items[0]:.4f}',
                    'cls': f'{loss_items[1]:.4f}',
                    'dfl': f'{loss_items[2]:.4f}'
                })

        val_map = 0.0
        if (epoch + 1) % 2 == 0:
            print(f"\nEvaluating validation mAP...")
            val_map = evaluate_epoch_map(model, val_loader, device, dataset_name="val")

        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_box = np.mean(train_box_losses)
        avg_train_cls = np.mean(train_cls_losses)
        avg_train_dfl = np.mean(train_dfl_losses)
        avg_val_box = np.mean(val_box_losses)
        avg_val_cls = np.mean(val_cls_losses)
        avg_val_dfl = np.mean(val_dfl_losses)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_bbox_loss'].append(avg_train_box)
        history['train_cls_loss'].append(avg_train_cls)
        history['train_dfl_loss'].append(avg_train_dfl)
        history['val_bbox_loss'].append(avg_val_box)
        history['val_cls_loss'].append(avg_val_cls)
        history['val_dfl_loss'].append(avg_val_dfl)
        history['train_mAP'].append(train_map)
        history['val_mAP'].append(val_map)

        # Learning rate scheduling
        scheduler.step()

        # Logging
        logging.info(
            f'Epoch {epoch+1}: '
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

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_dir / 'best.pt')
            logging.info(f'Best model saved with val loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    # Save final model and history
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
            model = YOLOv11(nc=1,ch=3)
        elif model_type == 'MASF-YOLO':
            model = MASF_YOLOv11(nc=1,ch=3)
        else:
            raise ValueError("Options are YOLOv11-s or MASF-YOLO")
        a_model, hist = train_single_run(seed, model, train_loader, val_loader, epochs, learning_rate, device, save_dir)
        list_of_models.append(a_model)
        list_of_histories.append(hist)

    analyzed_results = analyze_multi_run_training(list_of_histories)
    last_model = list_of_models[-1]

    return analyzed_results, last_model