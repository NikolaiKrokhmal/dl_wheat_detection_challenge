import torch.cuda
from models import Yolov1, FPNYolo, YOLOv1Loss
from data import get_all_dataloaders
from train import train_model_3_times
from test import test
from utils import analyze_multi_run_training
import pickle
import torch

if __name__ == "__main__":
    data_dir = './data/train/'
    csv_file = './data/train.csv'
    test_dir = './data/test/'
    SEED = [3, 7, 14]  # Danny Ric, Kimi Raikkonen, Fernando Alonso
    batch_size = 2
    epochs = 41
    img_size = 448
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########### Training and analyzing results of vanilla Yolov1 model
    # Set up training
    # train_loader, val_loader = get_all_dataloaders(
    #     data_dir=data_dir,
    #     csv_file=csv_file,
    #     grid_size=7,
    #     batch_size=batch_size,
    #     val_split=0.2,
    #     seed=SEED
    # )
    # criterion = YOLOv1Loss(grid_size=7, num_boxes=2, num_classes=1, lambda_coord=5, lambda_noobj=0.5)
    # t_yolov1 = train_model_3_times(model_type='yolov1',
    #                                train_loader=train_loader,
    #                                val_loader=val_loader,
    #                                random_seeds=SEED,
    #                                grid_size=7,
    #                                criterion=criterion,
    #                                learning_rate=learning_rate,
    #                                epochs=epochs,
    #                                device=device,
    #                                save_dir='runs')

    # # Running the model on the test images
    # t_yolov1 = Yolov1(7,2,1)
    # t_yolov1.load_state_dict(torch.load('runs/yolov1_s_7__seed_7.pt'))
    # test(t_yolov1, test_dir, conf_thresh=0.25)

    # # Plotting training graphs
    # hist_path = 'runs/yolov1_s_7_list_of_hist.pkl'
    # with open(hist_path, 'rb') as file:
    #     data = pickle.load(file)
    #     analyze_multi_run_training(data, 'yolov1')

    ######## Training and analyzing results of Yolov1 with FPN model and gridsize 7
    #    Set up training
    # train_loader, val_loader = get_all_dataloaders(
    #     data_dir=data_dir,
    #     csv_file=csv_file,
    #     grid_size=7,
    #     batch_size=batch_size,
    #     val_split=0.2,
    #     seed=SEED
    # )
    # criterion = YOLOv1Loss(grid_size=7, num_boxes=2, num_classes=1, lambda_coord=5, lambda_noobj=0.5)
    # t_fpn = train_model_3_times(model_type='fpn',
    #                             train_loader=train_loader,
    #                             val_loader=val_loader,
    #                             random_seeds=SEED,
    #                             grid_size=7,
    #                             criterion=criterion,
    #                             learning_rate=learning_rate,
    #                             epochs=epochs,
    #                             device=device,
    #                             save_dir='runs')

    # # Running the model on the test images
    # t_fpn = FPNYolo(7, 2, 1)
    # t_fpn.load_state_dict(torch.load('runs/fpn_s_7__seed_7.pt'))
    # test(t_fpn, test_dir, conf_thresh=0.25)

    # # Plotting training graphs
    # hist_path = 'runs/fpn_s_7_list_of_hist.pkl'
    # with open(hist_path, 'rb') as file:
    #     data = pickle.load(file)
    #     analyze_multi_run_training(data, 'fpn yolov1 s 7')


    ######## Training and analyzing results of Yolov1 with FPN model and gridsize 10
    #    Set up training
    # train_loader, val_loader = get_all_dataloaders(
    #     data_dir=data_dir,
    #     csv_file=csv_file,
    #     grid_size=10,
    #     batch_size=batch_size,
    #     val_split=0.2,
    #     seed=SEED
    # )
    # criterion = YOLOv1Loss(grid_size=10, num_boxes=2, num_classes=1, lambda_coord=4, lambda_noobj=0.1)
    # t_fpn = train_model_3_times(model_type='fpn',
    #                             train_loader=train_loader,
    #                             val_loader=val_loader,
    #                             random_seeds=SEED,
    #                             grid_size=10,
    #                             criterion=criterion,
    #                             learning_rate=learning_rate,
    #                             epochs=epochs,
    #                             device=device,
    #                             save_dir='runs')

    # # Running the model on the test images
    # t_fpn_10 = FPNYolo(10, 2, 1)
    # t_fpn_10.load_state_dict(torch.load('runs/fpn_s_10__seed_7.pt'))
    # test(t_fpn_10, test_dir, conf_thresh=0.25)

    # # Plotting training graphs
    # hist_path = 'runs/fpn_s_10_list_of_hist.pkl'
    # with open(hist_path, 'rb') as file:
    #     data = pickle.load(file)
    #     analyze_multi_run_training(data, 'fpn yolov1 s 10')



    ######## Training and analyzing results of Yolov1 with FPN model and gridsize 10
    #    Set up training
    # train_loader, val_loader = get_all_dataloaders(
    #     data_dir=data_dir,
    #     csv_file=csv_file,
    #     grid_size=14,
    #     batch_size=batch_size,
    #     val_split=0.2,
    #     seed=SEED
    # )
    # criterion = YOLOv1Loss(grid_size=14, num_boxes=2, num_classes=1, lambda_coord=4, lambda_noobj=0.05)
    # t_fpn = train_model_3_times(model_type='fpn',
    #                             train_loader=train_loader,
    #                             val_loader=val_loader,
    #                             random_seeds=SEED,
    #                             grid_size=14,
    #                             criterion=criterion,
    #                             learning_rate=learning_rate,
    #                             epochs=epochs,
    #                             device=device,
    #                             save_dir='runs')

    # # Running the model on the test images
    # t_fpn_14 = FPNYolo(14, 2, 1)
    # t_fpn_14.load_state_dict(torch.load('runs/fpn_s_14__seed_7.pt'))
    # test(t_fpn_14, test_dir, conf_thresh=0.25)

    # # Plotting training graphs
    # hist_path = 'runs/fpn_s_14_list_of_hist.pkl'
    # with open(hist_path, 'rb') as file:
    #     data = pickle.load(file)
    #     analyze_multi_run_training(data, 'fpn yolov1 s 14')
    #



