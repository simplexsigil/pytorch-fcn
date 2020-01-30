import datetime
import os
import os.path as osp
import shutil
import sys
from distutils.version import LooseVersion

import fcn
import numpy as np
import pytz
import skimage.io
import torch
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable

import torchfcn


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer, lr_scheduler,
                 train_loader, val_loader, out, max_epoch,
                 size_average=False, interval_val_viz=-1, epoch_callback_tuples=None):
        self.cuda = cuda

        self.cls_names = train_loader.dataset.class_names

        self.metric_history = []
        self.train_loss_history = []
        self.val_loss_history = []

        self.model = model
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.interval_val_viz = interval_val_viz

        if epoch_callback_tuples is None:
            epoch_callback_tuples = []
        else:
            self.epoch_callback_tuples = epoch_callback_tuples

        self.tmstamp_strt = datetime.datetime.now(pytz.timezone('Europe/Berlin'))
        self.size_average = size_average

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers_train = [
            'epoch',
            'iteration',
            'train/loss',
            'train/avg_acc',
            'train/avg_cls_wise_acc',
            'train/mean_iou',
            'train/fw_mean_iou'
            'elapsed_time'
        ]

        self.log_headers_val = [
            'epoch',
            'iteration',
            'valid/loss',
            'valid/avg_acc',
            'valid/avg_cls_wise_acc',
            'valid/mean_iou',
            'valid/fw_mean_iou',
            'elapsed_time'
        ]

        if not osp.exists(osp.join(self.out, 'log_train.csv')):
            with open(osp.join(self.out, 'log_train.csv'), 'w') as f:
                f.write(','.join(self.log_headers_train) + '\n')

        if not osp.exists(osp.join(self.out, 'log_val.csv')):
            with open(osp.join(self.out, 'log_val.csv'), 'w') as f:
                f.write(','.join(self.log_headers_val) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.best_mean_iou = 0
        self.best_epoch = 0

    def train(self):
        last_lr = self.lr_scheduler.get_last_lr()

        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch

            for call_epoch, callback in self.epoch_callback_tuples:
                if call_epoch == self.epoch:
                    callback()  # For example for unfreezing weight after some training.

            if self.epoch > 0 and self.interval_val_viz % self.epoch == 0:  # Print Current Learning rate.
                cur_lr = self.lr_scheduler.get_last_lr()

                if cur_lr != last_lr:
                    for par_group in self.optim.param_groups:
                        print("\nChanged param group learning rate to: {}".format(par_group["lr"]))
                    print("")

            assert self.model.training
            self.train_epoch()
            self.validate()

            self.lr_scheduler.step()

            if self.epoch >= self.max_epoch:
                break

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        assert self.model.training

        metrics = []
        train_loss = 0

        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                   desc='Train epoch=%d' % self.epoch, ncols=80, leave=False,
                                                   file=sys.stdout):
            iteration = batch_idx + self.epoch * len(self.train_loader)

            # Jump forward if resuming.
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue

            self.iteration = iteration

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.optim.zero_grad()

            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            train_loss += loss_data
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()

            self.optim.step()

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()

            acc, acc_cls, mean_iu, fwavacc = torchfcn.utils.label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)

            metrics.append((acc, acc_cls, mean_iu, fwavacc))

        metrics = np.mean(metrics, axis=0)

        train_loss /= len(self.train_loader)
        self.train_loss_history.append(train_loss)

        print("\rTrain epoch {ep}: Mean Pix Acc {acc:.4f} | Mean Class Acc {acc_cls:.4f} | "
              "Mean IoU {miu:.4f} | Weighted mean IoU {fwavacc:.4f}".format(ep=self.epoch,
                                                                            acc=metrics[0],
                                                                            acc_cls=metrics[1],
                                                                            miu=metrics[2],
                                                                            fwavacc=metrics[3]))

        with open(osp.join(self.out, 'log_train.csv'), 'a') as f:
            now = datetime.datetime.now(pytz.timezone('Europe/Berlin'))
            elpsd_time = (now - self.tmstamp_strt).total_seconds()
            log = [self.epoch, self.iteration, loss_data] + metrics.tolist() + [elpsd_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def validate(self):
        # Save training state to apply it again, later.
        training = self.model.training

        # Set model to evaluation mode.
        self.model.eval()

        # Preparing training variables.
        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        n_class = len(self.val_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                                                   desc='Validate epoch={}'.format(self.epoch), ncols=80, leave=False,
                                                   file=sys.stdout):
            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')

            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)

                if self.epoch % self.interval_val_viz == 0:
                    if len(visualizations) < 9:
                        viz = fcn.utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                        visualizations.append(viz)

        det_metrics = torchfcn.utils.label_accuracy_score_detailed(label_trues, label_preds, n_class, rt_cnf_mat=True)
        px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, cls_iou, cnf_mat = det_metrics

        metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class, det_metrics)

        self.metric_history.append(det_metrics)

        if len(visualizations) > 0:
            out = osp.join(self.out, 'visualization_viz')
            if not osp.exists(out):
                os.makedirs(out)
            out_file = osp.join(out, 'epoch%012d.jpg' % self.epoch)

            skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)
        self.val_loss_history.append(val_loss)

        print("\rValidate epoch {ep}: Mean Pix Acc {acc:.4f} | Mean Class Acc {acc_cls:.4f} | "
              "Mean IoU {miu:.4f} | Weighted mean IoU {fwavacc:.4f}".format(ep=self.epoch,
                                                                            acc=metrics[0],
                                                                            acc_cls=metrics[1],
                                                                            miu=metrics[2],
                                                                            fwavacc=metrics[3]))

        with open(osp.join(self.out, 'log_val.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Europe/Berlin')) - self.tmstamp_strt).total_seconds()
            log = [self.epoch, self.iteration, val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iou = metrics[2]
        is_best = mean_iou > self.best_mean_iou
        self.best_mean_iou = mean_iou if is_best else self.best_mean_iou
        self.best_epoch = self.best_epoch if is_best else self.epoch

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iou': self.best_mean_iou,
        }, osp.join(self.out, 'checkpoint.pth.tar'))

        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if self.epoch % self.interval_val_viz == 0:
            torchfcn.utils.plot_metrics(self.metric_history, self.train_loss_history, self.val_loss_history,
                                        cls_names=self.cls_names, best_idx=self.best_epoch, out_file=self.out)

        if training:
            self.model.train()
