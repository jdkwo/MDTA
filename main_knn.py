from __future__ import print_function

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import knn
from my_Dataset import MyDataset
import os
import sys
import argparse
import time
import math
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import sampler
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet, CustomGNNLayer, Disentanglement, recon_net, Discriminator
from losses import TripletLoss, MMDLoss
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,80',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='confidence threshold')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_false',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str,
                        default='save/SupCon/path_models/SupCE_path_resnet50_lr_0.001_decay_0'
                                '.0001_bsz_32_trial_0_cosine/last.pth',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'path':
        opt.n_cls = 6
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):

    if opt.dataset == 'path':
        s_train = MyDataset('H:/DFSData/cross_location/location_3/s_train', data_transform=False, test=0)
        val_dataset = MyDataset('H:/DFSData/cross_location/location_3/test', data_transform=False, test=1)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    s_train_loader = torch.utils.data.DataLoader(
        s_train, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return s_train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    recon = recon_net()
    gr = CustomGNNLayer()
    t_loss = TripletLoss()
    n_loss = TripletLoss()
    mmd_loss = MMDLoss()
    disentanglement = Disentanglement()
    discriminator = nn.ModuleDict({
        'c': Discriminator(out_num=opt.n_cls),
        'p': Discriminator(out_num=4),
        'o': Discriminator(out_num=5),
        'u': Discriminator(out_num=3)
    })
    knn_classifier = knn.MomentumQueue(1024, 1798, temperature=0.01, k=30, classes=6, eps_ball=1.1)
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        disentanglement = disentanglement.cuda()
        discriminator = discriminator.cuda()
        t_loss = t_loss.cuda()
        n_loss = n_loss.cuda()
        mmd_loss = mmd_loss.cuda()
        knn_classifier = knn_classifier.cuda()
        recon = recon.cuda()
        gr = gr.cuda()
    cudnn.benchmark = True

    return model, disentanglement, recon,  t_loss, n_loss, mmd_loss, discriminator, knn_classifier, gr


def reset_grad(optimizer):
    for _, opti in optimizer.items():
        opti.zero_grad()


def group_opt_step(optimizer, opt_keys):
    for k in opt_keys:
        optimizer[k].step()
    reset_grad(optimizer)


def recon_loss_minimizer(images, model, disentanglement, recon, optimizer):
    feature = model(images)
    feat_i, feat_p, feat_o, feat_u, feat_e = disentanglement(feature.detach())
    feat = torch.cat((feat_i, feat_p, feat_o, feat_u, feat_e), dim=1)
    recon_feat = recon(feat)
    loss = torch.sum((recon_feat - feature) ** 2) / (feature.shape[0] * feature.shape[1])
    loss.backward()
    group_opt_step(optimizer, ['dit', 'rec'])
    return loss, feat_i



def mutual_information_minimizer(images, model, disentanglement, n_loss, optimizer):
    feature = model(images)
    feat_i, feat_p, feat_o, feat_u, feat_e = disentanglement(feature.detach())
    feat = torch.cat((feat_i, feat_p, feat_o, feat_u, feat_e), dim=0)
    dim_len = len(feat_i)
    dim = torch.ones((dim_len * 5,), dtype=torch.int64)
    dim[0:dim_len] = 0
    dim[dim_len:dim_len * 2] = 1
    dim[dim_len * 2:dim_len * 3] = 2
    dim[dim_len * 3:dim_len * 4] = 3
    dim[dim_len * 4:dim_len * 5] = 4
    loss = n_loss(feat, dim) * 0.25
    reset_grad(optimizer)
    loss.backward()
    group_opt_step(optimizer, ['dit'])
    return loss


def discriminator_train(images, labels, position, orientation, user, model, disentanglement, discriminator, optimizer):
    feature = model(images)
    feat_i, feat_p, feat_o, feat_u, _ = disentanglement(feature)
    class_i = discriminator['c'](feat_i)
    loss_c = F.cross_entropy(class_i, labels)
    loss_p = F.cross_entropy(discriminator['p'](feat_p), position)
    loss_o = F.cross_entropy(discriminator['o'](feat_o), orientation)
    loss_u = F.cross_entropy(discriminator['u'](feat_u), user)
    loss = loss_p + loss_o + loss_u + loss_c
    reset_grad(optimizer)
    loss.backward()
    group_opt_step(optimizer, ['mod', 'dit', 'dis_c', 'dis_p', 'dis_o', 'dis_u'])
    return loss, class_i


def adversarial_train(images, labels, model, disentanglement, discriminator, optimizer, t_loss):
    feature = model(images)
    feat_i, _, _, _, _ = disentanglement(feature)
    loss_p = t_loss(discriminator['p'](feat_i), labels)
    loss_o = t_loss(discriminator['o'](feat_i), labels)
    loss_u = t_loss(discriminator['u'](feat_i), labels)
    loss = (loss_p + loss_o + loss_u) * 0.25
    reset_grad(optimizer)
    loss.backward()
    group_opt_step(optimizer, ['mod', 'dit'])
    return loss


def grap_train(model, knn_classifier, criterion, optimizer, opt):
        data = knn_classifier.build_graph()
        feat = model(data)
        loss_g = criterion(feat, knn_classifier.memory_labels)
        reset_grad(optimizer)
        loss_g.backward()
        group_opt_step(optimizer, ['gr'])
        return loss_g


def train(s_train_loader, model, disentanglement, recon,  t_loss, n_loss, discriminator,
            knn_classifier, optimizer, gr, epoch, opt):
    """one epoch training"""
    model.train()
    discriminator['p'].train()
    discriminator['o'].train()
    discriminator['u'].train()
    disentanglement.train()
    recon.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    feat = torch.empty((0, 1024), dtype=torch.float32).cuda()
    label = torch.empty((0,), dtype=torch.int64).cuda()

    end = time.time()
    for idx, (images, labels, position, orientation, user) in enumerate(s_train_loader):
        data_time.update(time.time() - end)
        s_images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        label = torch.cat((label, labels), dim=0)
        position = position.cuda(non_blocking=True)
        orientation = orientation.cuda(non_blocking=True)
        user = user.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(s_train_loader), optimizer)

        # compute loss
        loss_d, class_i = discriminator_train(s_images, labels, position, orientation, user, model, disentanglement, discriminator,
                                     optimizer)
        loss_a = adversarial_train(s_images, labels, model, disentanglement, discriminator, optimizer, t_loss)
        loss_m = mutual_information_minimizer(s_images, model, disentanglement, n_loss, optimizer)
        loss_r, feat_i = recon_loss_minimizer(s_images, model, disentanglement, recon, optimizer)
        feat = torch.cat((feat, feat_i), dim=0)
        losses1.update(loss_d.item(), bsz)
        losses2.update(loss_m.item(), bsz)
        acc1, acc5 = accuracy(class_i, labels, topk=(1, 5))

        top1.update(acc1[0], bsz)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(s_train_loader), batch_time=batch_time,
                data_time=data_time, loss1=losses1, loss2=losses2, top1=top1))
            sys.stdout.flush()
    knn_classifier.update_queue(feat, label)
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses1.avg, top1.avg


def validate(val_loader, model, disentanglement, knn_classifier,gr, epoch, opt, best_acc, bl):
    """validation"""
    model.eval()
    disentanglement.eval()
    knn_classifier.reduce_test()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    f_1 = []
    f_2 = []
    f_3 = []
    f_4 = []
    f_5 = []
    l = []
    p = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, position, orientation, user) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            # forward
            output = model(images)
            feat_i, feat_p, feat_o, feat_u, feat_e = disentanglement(output)
            if epoch == opt.epochs:
                for fe in torch.stack(list(feat_i)):
                    f_1.append(fe)
                for la in torch.stack(list(feat_p)):
                    f_2.append(la)
                for ma in torch.stack(list(feat_o)):
                    f_3.append(ma)
                for na in torch.stack(list(feat_u)):
                    f_4.append(na)
                for ea in torch.stack(list(feat_e)):
                    f_5.append(ea)
            for lab in torch.stack(list(labels)):
                l.append(lab)
            confidences, predicted = knn_classifier(feat_i, gr, opt.n_cls)
            predict = predicted[confidences >= opt.threshold]
            if predict.shape[0] > 0:
                knn_classifier.extend_test(feat_i[confidences >= opt.threshold], predict)
            pred_label = predicted
            for po in torch.stack(list(pred_label)):
                p.append(po)
            num = (pred_label == labels).sum().item()
            acc = num / bsz * 100.0
            top1.update(acc, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opt.print_freq == 0:
                print('memory_size: {}'.format(len(knn_classifier.memory_label)))
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    if top1.avg >= best_acc:
        bl = [p, l]
    if epoch == opt.epochs:
        f_1 = torch.stack(f_1)
        f_1 = f_1.view(-1, 1024)
        f_2 = torch.stack(f_2)
        f_2 = f_2.view(-1, 1024)
        f_3 = torch.stack(f_3)
        f_3 = f_3.view(-1, 1024)
        f_4 = torch.stack(f_4)
        f_4 = f_4.view(-1, 1024)
        f_5 = torch.stack(f_5)
        f_5 = f_5.view(-1, 1024)
        f = torch.cat((f_1, f_2, f_3, f_4, f_5), dim=0)
        dim_len = len(f_1)
        dim = torch.ones((dim_len * 5,), dtype=torch.int64)
        dim[0:dim_len] = 0
        dim[dim_len:dim_len * 2] = 1
        dim[dim_len * 2:dim_len * 3] = 2
        dim[dim_len * 3:dim_len * 4] = 3
        dim[dim_len * 4:dim_len * 5] = 4
        l = bl[0]
        l = torch.stack(l)
        l = l.view(-1)
        b = bl[1]
        b = torch.stack(b)
        b = b.view(-1)
        b = b.cpu()
        l = l.cpu()
        #start_tsne(f.cpu(), dim.cpu())
        draw_confusionMatrix(b, l)
    return losses.avg, top1.avg, bl


def start_tsne(x_train, y_train, shape=None):
    shape_list = ['o', 'x', '^', 's', 'p']
    color_list = ['black', 'red', 'gold', 'green', 'blue', 'silver']
    sh = []
    cl = []
    if shape is not None:
        for i in range(0, len(shape)):
            sh.append(shape_list[shape[i]])
            cl.append(color_list[y_train[i]])
    print("正在进行初始输入数据的可视化...")
    X_tsne = TSNE().fit_transform(x_train)
    plt.figure(figsize=(5, 5))
    if shape is not None:
        for i in range(0, len(y_train)):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=cl[i], marker=sh[i])
    else:
        for i in range(0, len(y_train)):
            cl.append(color_list[y_train[i]])
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cl)
    font_dict = dict(fontsize=14,
                     family='TimesNewRoman',
                     weight='normal')
    plt.xlabel('Dimension1', fontdict=font_dict)
    plt.ylabel('Dimension2', fontdict=font_dict)
    plt.show()


def draw_confusionMatrix(true_label, predict_label):
    label_names = ["Push", "Sweep", "Clap", "Slide", "Draw-O", "Draw-Z"]
    conf = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])
    sum = conf.sum(axis=1)
    confusion = conf / sum[:, np.newaxis]
    confusion = np.round(confusion, decimals=2)
    plt.matshow(confusion, cmap=plt.cm.Blues)  # Greens, Blues, Oranges, Reds
    plt.colorbar()
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            color = 'black'
            if i == j:
                color = 'white'
            plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center',
                         color=color)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(range(len(label_names)), label_names)
    plt.yticks(range(len(label_names)), label_names)
    plt.show()


def main():
    best_acc = 0
    best_epoch = 0
    opt = parse_option()
    # build data loader
    s_train_loader, val_loader = set_loader(opt)
    bl = []
    # build model and criterion
    model, disentanglement, recon, T_loss, N_loss, MMD_loss, discriminator, knn_classifier, gr = set_model(
        opt)
    # build optimizer
    optimizer = set_optimizer(opt, model, disentanglement, recon, discriminator)
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        for _, opti in optimizer.items():
            adjust_learning_rate(opt, opti, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(s_train_loader, model, disentanglement, recon, T_loss, N_loss, discriminator,
                                knn_classifier, optimizer, gr, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc, bl = validate(val_loader, model, disentanglement, knn_classifier, gr,
                                 epoch, opt, best_acc, bl)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        print('best accuracy: {:.2f}, best epoch : {}'.format(best_acc, best_epoch))

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, disentanglement, opt, save_file)


if __name__ == '__main__':
    main()
