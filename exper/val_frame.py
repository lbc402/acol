import sys
sys.path.append('../')

import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
import shutil
import my_optim
from models import *
import warnings
from utils import AverageMeter
from utils import Metrics
from utils.save_atten import SAVE_ATTEN
from utils.LoadData import data_loader
from utils.Restore import restore

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration',
             'Mass','Nodule','Pneumonia','Pneumothorax','Consolidation',
             'Edema','Emphysema','Fibrosis','Pleural_Thickening',
             'Hernia','No Finding']

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
DATA_DIR = '/'.join(os.getcwd().split('/')[:-2])+'/data'
print('Project Root Dir:',ROOT_DIR)

SNAPSHOT_DIR = os.path.join(ROOT_DIR,'snapshot_bins')

train_list = os.path.join(DATA_DIR, 'train_val_list.txt')
test_list = os.path.join(DATA_DIR, 'test_list.txt')

# Default parameters
LR = 0.001
EPOCH = 3
DISP_INTERVAL = 50

def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                        help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default=DATA_DIR,
                        help='Directory of training images')
    parser.add_argument("--train_list", type=str,
                        default=train_list)
    parser.add_argument("--test_list", type=str,
                        default=test_list)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--arch", type=str,default='vgg_v1')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", type=str, default='True')  # 恢复最后一个checkpoint训练
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default='')

    return parser.parse_args()


def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)

    model = torch.nn.DataParallel(model, range(args.num_gpu))
    model.cuda()

    optimizer = my_optim.get_optimizer(args, model)
    # checkpoint = torch.load(snapshot)
    # model.load_state_dict(checkpoint['state_dict'])

    if args.resume == 'True':
        restore(args, model, optimizer)

    return  model, optimizer

def val(args, model=None, current_epoch=0):

    outGT = torch.FloatTensor().cpu()
    out_A = torch.FloatTensor().cpu()   # branch A
    out_B = torch.FloatTensor().cpu()   # branch B

    if model is None:
        model, _ = get_model(args)
    model.eval()
    test_loader = data_loader(args, mode='test', sample=args.sample)

    print('Data count:', len(test_loader)*args.batch_size)
    print('Max iter:', len(test_loader))

    # save_atten = SAVE_ATTEN(save_dir='../save_bins/')


    for idx, dat in tqdm(enumerate(test_loader)):
        img_path, img, label_in = dat

        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label_input = label_in.repeat(10, 1)
            label = label_input.view(-1)
        else:
            label = label_in

        img_var, label_var = img.cuda(), label.cuda()

        with torch.no_grad():
            logits = model(img_var, label_var, mode='val')

        logits0 = logits[0]
        logits1 = logits[1]

        if args.tencrop == 'True':
            logits0 = logits0.view(bs, ncrops, -1).mean(1)
            logits1 = logits1.view(bs, ncrops, -1).mean(1)

        outGT = torch.cat((outGT, label.cpu().data), 0)
        out_A = torch.cat((out_A, logits0.cpu().data), 0)
        out_B = torch.cat((out_B, logits1.cpu().data), 0)

        # model.module.save_erased_img(img_path)
        last_featmaps = model.module.get_localization_maps()
        np_last_featmaps = last_featmaps.cpu().data.numpy()

        # Save 100 sample masked images by heatmaps
        # if idx < 100/args.batch_size:
        #     save_atten.get_masked_img(img_path, np_last_featmaps, label_in.numpy(), size=(0,0), maps_in_dir=False)

        # save_atten.get_masked_img(img_path, np_last_featmaps, label_in.numpy(),size=(0,0),
        #                           maps_in_dir=True, save_dir='../heatmaps',only_map=True )

        # np_scores, pred_labels = torch.topk(logits0,k=args.num_classes,dim=1)
        # pred_np_labels = pred_labels.cpu().data.numpy()
        # save_atten.save_top_5_pred_labels(pred_np_labels[:,:5], img_path, idx)
        # # pred_np_labels[:,0] = label.cpu().numpy() #replace the first label with gt label
        # # save_atten.save_top_5_atten_maps(np_last_featmaps, pred_np_labels, img_path)


    auc_A = Metrics.compute_AUCs(outGT, out_A, args.num_classes)
    print(dict(zip(label_list, auc_A)))
    auc_B = Metrics.compute_AUCs(outGT, out_B, args.num_classes)
    print(dict(zip(label_list, auc_B)))


    # save_name = os.path.join(SNAPSHOT_DIR, 'test_result.txt')
    # with open(save_name, 'a') as f:
    #     f.write('%.3f'% auc_A)


if __name__ == '__main__':
    args = get_arguments()
    # import json
    # print('Running parameters:\n')
    # print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    val(args)
