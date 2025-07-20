import torch.cuda
from model import Yolov1, YOLOv1Loss
from data import get_all_dataloaders
from train import train


if __name__ == "__main__":
    root_dir = './'
    data_dir = './data/train/'
    csv_file = './data/train.csv'
    SEED = 42
    batch_size = 4
    epochs = 2
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

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = YOLOv1Loss(7, 2, 1)

    # Train
    train(model, train_loader, val_loader, optimizer, criterion, epochs, device)