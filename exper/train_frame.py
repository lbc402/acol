import sys
sys.path.append('../')
import torch
import argparse
import os
import time
import shutil
import json
import datetime
import my_optim
import warnings
from tqdm import tqdm
from models import *
from utils import AverageMeter
from utils import Metrics
from utils.LoadData import data_loader
from utils.Restore import restore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

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
LR = 0.0001
EPOCH = 10
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
    parser.add_argument("--resume", type=str, default='False')  # 恢复最后一个checkpoint训练
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = eval(args.arch).model(pretrained=False,
                                  num_classes=args.num_classes,
                                  threshold=args.threshold,
                                  args=args)
    model.cuda()
    # 并行语句
    model = torch.nn.DataParallel(model, range(args.num_gpu))

    optimizer = my_optim.get_adam(args, model)

    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return  model, optimizer


def train(args):
    # torch.backends.cudnn.deterministic = True

    batch_time = AverageMeter()
    losses = AverageMeter()

    writer = SummaryWriter(log_dir='acol_nih')

    model, optimizer= get_model(args)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True, min_lr=1e-7)

    model.train()

    train_loader, val_loader = data_loader(args, mode='train', sample=args.sample)
    _, input_data, input_label = next(iter(train_loader))
    writer.add_graph(model, (input_data, input_label))

    # with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
    #     config = json.dumps(vars(args), indent=4, separators=(',', ':'))
    #     fw.write(config)
    #     fw.write('#epoch, loss\n')


    total_epoch = args.epoch
    global_counter = args.global_counter

    end = time.time()
    max_iter = total_epoch*len(train_loader)
    print('Max iter:', max_iter)

    best_loss = 9999

    outGT = torch.FloatTensor().cpu()
    outPRED = torch.FloatTensor().cpu()
    outPRED1 = torch.FloatTensor().cpu()

    for epoch in range(total_epoch):
        model.train()
        losses.reset()
        batch_time.reset()

        steps_per_epoch = len(train_loader)
        # print('learning rate: %g' % args.lr)

        for idx, (_, img, label) in enumerate(train_loader):
            global_counter += 1
            img, label = img.cuda(), label.cuda()
            logits = model(img, label) # automatically call model.forward()
            loss, loss_cls, loss_cls_ers = model.module.get_loss(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.detach())

            batch_time.update(time.time() - end)

            end = time.time()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA(Estimated Time of Arrival)
                niter = global_counter / args.disp_interval
                eta_seconds = ((total_epoch - epoch)*steps_per_epoch + (steps_per_epoch - idx))*batch_time.avg
                eta_str = datetime.timedelta(seconds=int(eta_seconds))
                eta_seconds_epoch = steps_per_epoch*batch_time.avg
                eta_str_epoch = datetime.timedelta(seconds=int(eta_seconds_epoch))
                print('Epoch: [{0}][{1}/{2}][{3}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ETA: {eta_str}({eta_str_epoch})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_A: {loss_A:.4f} Loss_B: {loss_B:.4f}'
                      .format(
                    epoch, global_counter%len(train_loader), len(train_loader),global_counter, batch_time=batch_time,
                    eta_str=eta_str, eta_str_epoch = eta_str_epoch, loss=losses, loss_A=loss_cls, loss_B=loss_cls_ers))
                writer.add_scalar('loss', loss, niter)
                writer.add_scalar('loss_A',loss_cls, niter)
                writer.add_scalar('loss_B',loss_cls_ers, niter)


        # validate and visualize
        val_loss = val(model, outGT, outPRED, outPRED1, val_loader)

        if(val_loss < best_loss):
            best_loss = val_loss
            save_checkpoint(args,
                {
                    'epoch': epoch,
                    'arch': args.arch,            # archive
                    'global_counter': global_counter,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict()
                }, is_best=False,
                filename='%s_epoch_%d_glo_step_%d.pth.tar'
                         %(args.dataset, epoch, global_counter))

        scheduler.step(val_loss)

        # with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        #     fw.write('%d,%.4f\n'%(epoch, losses.avg))

    # auc = Metrics.compute_AUCs(outGT, outPRED, args.num_classes)
    # print('auc: %f' % auc)

    writer.export_scalars_to_json("./test.json")
    writer.close()


def val(model, outGT, outPRED, outPRED1, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    loss_list = []
    loss_cls_list = []
    loss_ers_list = []

    for i, (_, val_input, target) in tqdm(enumerate(dataloader)):
        val_input, target = val_input.cuda(), target.cuda()
        with torch.no_grad():
            logits = model(val_input, target, mode='val') # automatically call model.forward()
            loss, loss_cls, loss_ers = model.module.get_loss(logits, target)

        outGT = torch.cat((outGT, target.cpu().data), 0)
        outPRED = torch.cat((outPRED, logits[0].cpu().data), 0)
        outPRED1 = torch.cat((outPRED1, logits[1].cpu().data), 0)

        loss_list.append(loss.detach().cpu())
        loss_cls_list.append(loss_cls.detach().cpu())
        loss_ers_list.append(loss_ers.detach().cpu())

    val_loss =  sum(loss_list)/(1.0*len(loss_list))
    val_loss_cls = sum(loss_cls_list)/(1.0*len(loss_cls_list))
    val_loss_ers = sum(loss_ers_list)/(1.0*len(loss_ers_list))

    print('Val_Loss {loss:.4f} \t'
              'Val_Loss_A: {loss_A:.4f} Val_Loss_B: {loss_B:.4f}'
                  .format(loss=val_loss, loss_A=val_loss_cls, loss_B=val_loss_ers))

    auc = Metrics.compute_AUCs(outGT, outPRED, args.num_classes)
    print(dict(zip(label_list, auc)))
    auc = Metrics.compute_AUCs(outGT, outPRED1, args.num_classes)
    print(dict(zip(label_list, auc)))

    return val_loss

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    # print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
