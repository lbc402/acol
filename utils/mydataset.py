import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import cv2
import pandas as pd

random.seed(42)
label_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration',
             'Mass','Nodule','Pneumonia','Pneumothorax','Consolidation',
             'Edema','Emphysema','Fibrosis','Pleural_Thickening',
             'Hernia','No Finding']

class dataset(Dataset):

    """
    训练集和验证集的图像名称写在同一个文件中:train_val_list.txt
    测试集的图像名称写在:test_list.txt
    标签写在:Data_Entry_2017.csv
    """

    def __init__(self, data_dir, transform=None, mode='train', sample=0):

        self.transform = transform
        self.label_csv = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))

        if mode=='test':
            imgs = []
            self.data_dir = os.path.join(data_dir, 'test')
            with open(os.path.join(data_dir, 'test_list.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgs.append(os.path.join(self.data_dir, line.rstrip()))
        else:
            self.data_dir = os.path.join(data_dir, 'train')
            imgs = [os.path.join(self.data_dir,img) for img in os.listdir(self.data_dir)]

        if sample > 0:
            imgs = random.sample(imgs, int(sample))  # 抽样

        if mode != 'test':
            random.shuffle(imgs)

        imgs_num = len(imgs)

        if mode=='test':
            self.imgs = imgs
        elif mode=='train':
            self.imgs = imgs[:int(0.8*imgs_num)]
        else:

            self.imgs = imgs[int(0.8*imgs_num):]   # validation set


    def __getitem__(self, index):

        img_path = self.imgs[index]
        image = Image.open(img_path).convert('RGB')  # 注意：nih数据集是灰度图像

        img_name = img_path.split('/')[-1]
        labels = self.label_csv.loc[self.label_csv['Image Index']==img_name, 'Finding Labels'].values[0]
        labels = labels.strip().split('|')
        label = np.zeros(14)
        for l in labels:
            if l != 'No Finding':
                idx = label_list.index(l)
                label[idx] = 1

        if self.transform is not None:
            image = self.transform(image)


        return img_path, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.imgs)







class dataset_with_mask(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, self.get_name_id(self.image_list[idx])+'.png')
        mask = cv2.imread(mask_name)
        mask[mask==0] = 255
        mask = mask - 1
        mask[mask==254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]


    def get_name_id(name_path):
        name_id = name_path.strip().split('/')[-1]
        name_id = name_id.strip().split('.')[0]
        return name_id

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)


if __name__ == '__main__':
    # datalist = '/data/zhangxiaolin/data/INDOOR67/list/indoor67_train_img_list.txt';
    # data_dir = '/data/zhangxiaolin/data/INDOOR67/Images'
    # datalist = '/data/zhangxiaolin/data/STANFORD_DOG120/list/train.txt';
    # data_dir = '/data/zhangxiaolin/data/STANFORD_DOG120/Images'
    datalist = '../datalist/CUB/test_list.txt'
    # data_dir = '/data/zhangxiaolin/data/VOC2012'
    # datalist = '../../image/train_val_list.txt';
    data_dir = '../../image/train'

    data = dataset(datalist, data_dir)
    print(len(data.label_list))

    img_mean = np.zeros((len(data), 3))
    img_std = np.zeros((len(data), 3))
    for idx in range(len(data)):
        img, _ = data[idx]
        numpy_img = np.array(img)
        per_img_mean = np.mean(numpy_img, axis=(0,1))/255.0
        per_img_std = np.std(numpy_img, axis=(0,1))/255.0

        img_mean[idx] = per_img_mean
        img_std[idx] = per_img_std

    print(np.mean(img_mean, axis=0), np.mean(img_std, axis=0))

