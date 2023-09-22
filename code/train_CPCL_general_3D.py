import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la_heart import (LA_heart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA_data_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_CPCL_general_staLoss', help='experiment_name') # _staLoss: --stability_loss 1
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=6,
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=80,
                    help='total samples')

# pretrain
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--PAweight', type=float,
                    default=10, help='prototypical weight')
parser.add_argument('--stability_loss', type=int,
                    default=1, help='whether use stability loss')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def getPrototype(features, mask, class_confidence):
    # adjust the features H, W shape
    fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')  # 3D uses tri, 2D uses bilinear
    # print(fts.shape)
    # print(mask.shape)
    # masked average pooling
    mask_new = mask.unsqueeze(1)  # bs x 1 x Z x H x W
    # get the masked features
    masked_features = torch.mul(fts, mask_new)  # here use a broadcast mechanism: https://blog.csdn.net/da_kao_la/article/details/87484403
    masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) / ((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
    # print(sum1.shape)
    # print('masked fts:', masked_fts.shape)
    return masked_fts


def calDist(fts_adj_size, prototype):
    prototype_new = prototype.unsqueeze(2)
    prototype_new = prototype_new.unsqueeze(3)
    prototype_new = prototype_new.unsqueeze(4)
    scalar = 20
    dist = F.cosine_similarity(fts_adj_size, prototype_new, dim=1) * scalar
    return dist

def prototype_pred(src_prototypes, feature_tgt, mask_src, class_nums):
    # 1 extract the foreground features via masked average pooling
    feature_tgt_adj = F.interpolate(feature_tgt, size=mask_src.shape[-3:], mode='trilinear')  # 3D uses tri, 2D uses bilinear, [2, 256, 96, 96, 96]
    # print('feature_tgt_adj:', feature_tgt_adj.shape)
    for class_index in range(class_nums):
        dist = calDist(feature_tgt_adj, src_prototypes[class_index]).unsqueeze(1)
        final_dist = dist if class_index == 0 else torch.cat((final_dist, dist), 1)
    final_dist_soft = torch.softmax(final_dist, dim=1)
    return final_dist_soft

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    pretrain_model = args.pretrain_model
    PAweight = args.PAweight

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    if pretrain_model:
        model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")
    ema_model = create_model(ema=True)

    db_train = LA_heart(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # define the labeled data list
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_batch = label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]                    

            # noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch # here we do not use noise perturbation

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            print('model output shape:', outputs.shape)
            if args.stability_loss:
                ema_inputs_stability = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    ema_output_stability = ema_model(ema_inputs_stability)
                MT_stability_loss = torch.mean((ema_output_stability - outputs) ** 2)
            else:
                MT_stability_loss = 0

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
            
            # labeled-to-unlabeled process
            label_fts = model.featuremap_center[:args.labeled_bs]
            unlabel_fts = ema_model.featuremap_center 
            print('label data feature map shape:', label_fts.shape)  
            label_fts_confidence = []
            for class_index in range(num_classes):
                label_confidence = outputs_soft[:args.labeled_bs, class_index, :, :, :].unsqueeze(1)
                label_fts_confidence.append(label_confidence)
            
            label_fts_prototypes = []
            for class_index in range(num_classes):
                label_fts_proto = getPrototype(label_fts, (label_batch[:args.labeled_bs]==class_index), label_fts_confidence[class_index]).detach()
                label_fts_prototypes.append(label_fts_proto)
            
            # prototype-based labeled-to-unlabeled prediction
            labeltounlabel_pred = prototype_pred(label_fts_prototypes, unlabel_fts, label_batch[:args.labeled_bs], num_classes)
            L2U_consistency_loss = torch.mean((labeltounlabel_pred - ema_output_soft) ** 2)

            # unlabeled-to-labeled process
            unlabel_preMask = torch.argmax(ema_output_soft, dim=1) # [bs/2, 96, 96, 96]
            unlabel_fts_confidence = []
            for class_index in range(num_classes):
                unlabel_confidence = ema_output_soft[:, class_index, :, :, :].unsqueeze(1)
                unlabel_fts_confidence.append(unlabel_confidence)
            
            unlabel_fts_prototypes = []
            for class_index in range(num_classes):
                unlabel_fts_proto = getPrototype(unlabel_fts, (unlabel_preMask==class_index), unlabel_fts_confidence[class_index]).detach()
                unlabel_fts_prototypes.append(unlabel_fts_proto)
            
            # prototype-based unlabeled-to-labeled prediction
            unlabeltolabel_pred = prototype_pred(unlabel_fts_prototypes, label_fts, unlabel_preMask, num_classes)
            U2L_loss = ce_loss(unlabeltolabel_pred, label_batch[:args.labeled_bs][:])

            # supervised loss
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            protoalign_weight = PAweight * get_current_consistency_weight(iter_num // 150)


            loss = supervised_loss + consistency_weight * (L2U_consistency_loss + MT_stability_loss) + protoalign_weight * U2L_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/L2U_loss',
                              L2U_consistency_loss, iter_num)
            writer.add_scalar('info/U2L_loss',
                              U2L_loss, iter_num)
            writer.add_scalar('info/MT_stability_loss',
                              MT_stability_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 200 and iter_num % 100 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (
                    iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
