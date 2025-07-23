import torch.cuda
from model import Yolov1, FPNYolo
from data import get_all_dataloaders
from train import train
from test import test


if __name__ == "__main__":
    data_dir = './data/train/'
    csv_file = './data/train.csv'
    test_dir = './data/test/'
    SEED = 159
    batch_size = 16
    epochs = 30
    img_size = 448
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloaders
    train_loader, aug_loader, val_loader = get_all_dataloaders(
        data_dir=data_dir,
        csv_file=csv_file,
        batch_size=batch_size,
        val_split=0.2,
        seed=SEED
    )

    # Model
    model = Yolov1(7, 2, 1).to(device)

    # Train
    train(model, train_loader, val_loader, learning_rate, epochs, device)

    # Test model
    test(model, test_dir, conf_thresh=0.25)
