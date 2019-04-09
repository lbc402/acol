import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import os
import cv2
import numpy as np


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, args=None, threshold=0.6, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.cls = self.classifier(512, num_classes)
        self.cls_erase = self.classifier(512, num_classes)
        self.onehot = args.onehot
        self.sigmoid = nn.Sigmoid()

        self.threshold = threshold

        if init_weights:
            self._initialize_weights()
        #Optimizer
        if args is not None and args.onehot=='True':
            self.loss_cross_entropy = nn.BCELoss(weight=None, size_average=True, reduce=True)
        else:
            self.loss_cross_entropy = nn.CrossEntropyLoss()

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1), #fc6
            # nn.BatchNorm2d(v),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  #fc8
        )

    def forward(self, x, label=None, mode='train'):
        self.img_erased = x
        # Backbone
        x = self.features(x)
        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Branch A
        out = self.cls(feat)
        self.map1 = out
        logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)  # GAP

        # if not self.training:
        #     _, label = torch.max(logits_1, dim=1)

        # erase feature maps

        localization_map_normed = self.get_atten_map(logits_1, out, label, True, mode)
        self.attention = localization_map_normed
        feat_erase = self.erase_feature_maps(localization_map_normed, feat, self.threshold)

        # Branch B
        out_erase = self.cls_erase(feat_erase)
        self.map_erase = out_erase
        logits_ers = torch.mean(torch.mean(out_erase, dim=2), dim=2)

        return [self.sigmoid(logits_1), self.sigmoid(logits_ers)]

    # modify by myself, feature mapping is extracted according to label and weighted by multiple labels
    def get_atten_map(self, pred_value, feature_maps, gt_labels, normalize=True, mode='train'):
        labels = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = atten_map.cuda()
        for batch_idx in range(batch_size):

            if mode == 'train' or mode == 'location':
                pred = self.sigmoid(pred_value).cpu().data.numpy()
                temp = torch.zeros([1,1,feature_map_size[2], feature_map_size[3]]).cuda()
                label = labels.data[batch_idx].cpu().data.numpy()
                label_index = np.where(label==1)[0]   # get index
                for i in label_index:
                    temp = temp.add_(feature_maps[batch_idx, i, :,:])

                if len(label_index) > 0:
                    temp = temp / len(label_index)   # mean

                atten_map[batch_idx,:,:] = torch.squeeze(temp)
            else:
                temp = torch.zeros([1, 1, feature_map_size[2], feature_map_size[3]]).cuda()
                label = labels.data[batch_idx].cpu().data.numpy()
                label_index = np.where(label == 1)[0]
                if len(label_index) == 0:
                    atten_map[batch_idx, :, :] = torch.squeeze(temp)
                else:
                    # pred_value = self.sigmoid(pred_value)
                    # pred = pred_value.data[batch_idx].cpu().data.numpy()
                    # pred_index = np.where(pred == np.max(pred))[0][0]   # 取输出最大的那个特征映射
                    atten_map[batch_idx,:,:] = feature_maps[batch_idx, label_index, :,:]


        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

    # GAP 后的 CAM
    def get_val_atten_map(self, pred_value, feature_maps):
        pred_value = self.sigmoid(pred_value)
        pred = pred_value.data[batch_idx]
        pred = torch.unsqueeze(torch.unsqueeze(pred, 1), 1)
        t = torch.squeeze(self.map1[batch_idx, :,:,:]) * pred
        atten_map[batch_idx,:,:] = torch.sum(t, 0, keepdim=True)

    # multi-labels for multi-feature_maps
    def get_multi_atten_map(self, feature_maps, gt_labels):
        labels = gt_labels.long()
        label_list = []

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        for batch_idx in range(batch_size):
            label = labels.data[batch_idx].cpu().data.numpy()
            label_index = np.where(label==1)[0]   # get index
            label_list.append(label_index)   # 2-dim list

        return label_list

    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):
        # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        # atten_map_normed = self.up_resize(atten_map_normed)
        if len(atten_map_normed.size())>3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        pos = torch.ge(atten_map_normed, threshold)
        mask = torch.ones(atten_shape).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        #erase
        erased_feature_maps = feature_maps * mask

        return erased_feature_maps

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def add_heatmap2img(self, img, heatmap):
        # assert np.shape(img)[:3] == np.shape(heatmap)[:3]

        heatmap = heatmap * 255
        heatmap = heatmap.astype(np.uint8)
        # .astype(np.uint8)
        color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return img_res

    def get_loss(self, logits, gt_labels):
        if self.onehot == 'True':
            gt = gt_labels.float()
        else:
            gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits[0], gt)
        loss_cls_ers = self.loss_cross_entropy(logits[1], gt)

        loss_value = loss_cls + loss_cls_ers

        return loss_value, loss_cls, loss_cls_ers

    # return the max value in every dims
    def get_localization_maps(self, concat=True):
        map1 = self.normalize_atten_maps(self.map1)
        map_erase = self.normalize_atten_maps(self.map_erase)
        if concat == True:
            return torch.max(map1, map_erase)
        else:
            return map_erase

    def get_heatmaps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1,]

    def get_fused_heatmap(self, gt_label):
        maps = self.get_heatmaps(gt_label=gt_label)
        fuse_atten = maps[0]
        return fuse_atten

    def get_maps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1, ]


    def save_erased_img(self, img_path, img_batch=None):
        mean_vals = [0.485, 0.456, 0.406]   # imagenet parameter
        std_vals = [0.229, 0.224, 0.225]
        if img_batch is None:
            img_batch = self.img_erased
        if len(img_batch.size()) == 4:
            batch_size = img_batch.size()[0]
            for batch_idx in range(batch_size):
                imgname = img_path[batch_idx]
                nameid = imgname.strip().split('/')[-1].strip().split('.')[0]

                # atten_map = F.upsample(self.attention.unsqueeze(dim=1), (321,321), mode='bilinear')
                mask = F.upsample(self.attention.unsqueeze(dim=1), (256,256), mode='bilinear')
                # atten_map = F.upsample(self.attention, (224,224), mode='bilinear')
                # mask = F.sigmoid(20*(atten_map-0.5))

                mask = mask.cpu().data.numpy()
                mask = mask[batch_idx].transpose((1,2,0))
                # mask = mask.repeat([3], axis=2)
                mask = mask[:,:,0]


                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255

                # mask = cv2.resize(mask, (321,321))
                img_dat = self.add_heatmap2img(img_dat, mask)
                save_path = os.path.join('../erase_figs/', nameid+'.png')
                cv2.imwrite(save_path, img_dat)

    def save_finally_map(self, img_path, img_batch=None):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# return value is used to be self.feature in VGG
def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# define model channels
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    # 'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# define dilation parameters
dilation = {
    # 'D_deeplab': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 2, 2, 2, 'N'],
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}


def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D1'], dilation=dilation['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], dilation=dilation['D1'], batch_norm=True), **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model