# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset, dataset_with_mask
import torchvision
import torch
import numpy as np

def data_loader(args, mode='train', sample=0):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    elif 'imagenet' in args.dataset:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
    else:
        mean_vals = [0.505, 0.505, 0.505]   # nih dataset parameter
        std_vals = [0.252, 0.252, 0.252]


    input_size = int(args.input_size)


    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    tsfm_test = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals)
                                    ])

    test_dataset = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.TenCrop(224),
                                    transforms.Lambda
                                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    transforms.Lambda
                                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                ])


    if mode == 'train':
        img_train = my_dataset(data_dir=args.img_dir,
                               transform=tsfm_train, mode='train', sample=sample)
        img_val = my_dataset(data_dir=args.img_dir,
                              transform=tsfm_test, mode='val', sample=sample)

        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(img_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        return train_loader, val_loader
    else:
        img_test = my_dataset(data_dir=args.img_dir,
                               transform=tsfm_test, mode='test', sample=sample)
        test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        return test_loader

