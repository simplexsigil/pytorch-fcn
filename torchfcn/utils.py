import os
import os.path as osp
import shlex
import subprocess

import fcn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sn
import skimage
import torch
from torch.autograd import Variable

import torchfcn
from torchfcn.trainer import cross_entropy2d


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class, det_metrics=None):
    """Returns accuracy score evaluation result.

      - average pixelwise accuracy
      - mean classwise accuracy (not the same, if classes have different frequency)
      - mean Intersection over Union
      - Frequency Weighted Averaged Accuracy
    """
    if det_metrics is None:
        px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, iou, _ = \
            label_accuracy_score_detailed(label_trues, label_preds, n_class)
    else:
        px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, iou, _ = det_metrics
    # Average pixel wise accuracy
    avg_acc = tr_all / px_all

    # Average class wise accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_cls_acc = tr_cls_wise / cls_pixel_counts

    avg_cls_acc = np.nanmean(avg_cls_acc)

    mean_iou = np.nanmean(iou)

    # Frequency weighted mean Intersection over Union
    freq = cls_pixel_counts / px_all
    fw_mean_iou = (freq[freq > 0] * iou[freq > 0]).sum()
    return avg_acc, avg_cls_acc, mean_iou, fw_mean_iou


def label_accuracy_score_detailed(label_trues, label_preds, n_class, rt_cnf_mat=False):
    """
    Returns metrics given true and predicted labels of an image segmentation.
    :param label_trues: Segmentation ground truths of shape (batch_size, height, width).
    :param label_preds: Segmentation predictions of shape (batch_size, height, width).
    :param n_class: Number of classes.
    :param rt_cnf_mat: If true, a confusion matrix is returned as last parameter, None otherwise.
    :return: px_all (All pixel count), tr_all (All correctly classified pixel count),
    tr_cls_wise (Class wise correctly classified pixel count),
    cls_pixel_counts (True pixel count per class),
    cls_clsfd_pixel_counts (Classwise classified pixel count - TP and FP),
    iou (Intersection over Union - TP / TP + FP + FN),
    cnf_mat (Confusion matrix)
    """
    # Histogram / Confusion Matrix (true classes)x(predicted classes)
    cnf_mat = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        cnf_mat += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    tr_cls_wise = np.diag(cnf_mat)  # Class wise true positive pixels.
    tr_all = tr_cls_wise.sum()  # All true positive pixels.
    px_all = cnf_mat.sum()  # All pixels.

    cls_pixel_counts = cnf_mat.sum(axis=1)  # All pixels belonging to class.
    cls_clsfd_pixel_counts = cnf_mat.sum(axis=0)  # All pixels classified to class.

    # Intersection over Union: TP / (TP + FP + FN).
    # Per segmentation class:
    # correctly segmented pixels /
    # (correctly pixels +
    # pixels which are assigned to this class but do not belong +
    # pixels which do belong but are not assigned)
    with np.errstate(divide='ignore', invalid='ignore'):
        cls_iou = tr_cls_wise / (cls_pixel_counts + cls_clsfd_pixel_counts - tr_cls_wise)

    return px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, cls_iou, cnf_mat if rt_cnf_mat else None


def plot_metrics(metric_history, train_los_history, val_los_history, cls_names, best_idx, out_file):
    # px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, cls_iou, cnf_mat = metric_history[0]
    hist_len = len(metric_history)

    px_all_hist = np.zeros(hist_len)  # History of all pixel counts (actually this should always be the same)
    tr_all_hist = np.zeros(hist_len)  # History of all true pixel counts (actually this should always be the same)
    tr_cls_wise_hist = np.zeros((len(cls_names), hist_len))  # History of class wise true pixel counts
    cls_clsfd_pixel_counts_hist = np.zeros((len(cls_names), hist_len))  # History of class wise classified pixel counts
    cls_pixel_counts_hist = np.zeros((len(cls_names), hist_len))  # History of class wise pixel counts
    cls_iou_hist = np.zeros((len(cls_names), hist_len))  # History of class wise intersection over union
    cnf_mat_hist = np.zeros((len(cls_names), len(cls_names), hist_len))  # History of confusion matrices

    for idx, metrics in enumerate(metric_history):
        px_all, tr_all, tr_cls_wise, cls_pixel_counts, cls_clsfd_pixel_counts, cls_iou, cnf_mat = metrics

        px_all_hist[idx] = px_all
        tr_all_hist[idx] = tr_all
        tr_cls_wise_hist[:, idx] = tr_cls_wise
        cls_clsfd_pixel_counts_hist[:, idx] = cls_clsfd_pixel_counts
        cls_pixel_counts_hist[:, idx] = cls_pixel_counts
        cls_iou_hist[:, idx] = cls_iou
        cnf_mat_hist[:, :, idx] = cnf_mat

    acc_hist = tr_all_hist.astype(float) / px_all_hist

    cnf_fig = plt.figure("Classwise Normalized Confusion Matrix", figsize=[10, 7])
    plt.title('Normalized Confusion Matrix of Epoch {}'.format(best_idx))
    with np.errstate(divide='ignore', invalid='ignore'):
        # Only show last confusion matrix for the moment.
        cm_normalized_cls = cnf_mat_hist[:, :, best_idx].astype(float) / cls_pixel_counts_hist[:, best_idx].sum(axis=0)

    sn.heatmap(cm_normalized_cls, annot=True, fmt="3.2f", xticklabels=cls_names, yticklabels=cls_names, cmap="YlGnBu",
               linewidths=.5)
    plt.savefig(osp.join(out_file, 'cnf_mat.pdf'), bbox_inches="tight", pad_inches=0.3)
    plt.close(cnf_fig)

    train_val_fig = plt.figure("Train Loss / Val Loss / Val Accuracy", figsize=[10, 7])
    plt.title('Train loss / Val loss / Val Accuracy')

    ax0 = train_val_fig.axes[0]
    ax1 = ax0.twinx()

    ax0.plot(train_los_history, color='red', label="Train Loss")
    ax0.plot(val_los_history, color='green', label="Val Loss")
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')

    ax1.plot(acc_hist, color='blue', label="Val Accuracy")
    ax1.set_ylabel('Accuracy')

    ax0.set_xticks(range(hist_len))
    ax0.set_ylim(0, max(max(train_los_history), max(val_los_history)))

    ax0.legend(loc="center left")
    ax1.legend(loc="center right")
    plt.savefig(osp.join(out_file, 'train_val_loss.pdf'), bbox_inches="tight", pad_inches=0.3)
    plt.close(train_val_fig)

    iou_fig = plt.figure("Classwise Intersection over Union", figsize=[10, 7])
    for idx, cls in enumerate(cls_names):
        plt.plot(cls_iou_hist[idx], label="IoU {}".format(cls))
    plt.xlabel('Epoch')
    plt.ylabel('Intersection over Union')
    plt.title('Intersection over Union per Class')
    plt.legend()
    plt.savefig(osp.join(out_file, 'iou_class.pdf'), bbox_inches="tight", pad_inches=0.3)
    plt.close(train_val_fig)
    plt.close(iou_fig)


def test(model, dataloader, use_cuda):
    was_training = model.training
    model.eval()

    cls_names = dataloader.dataset.class_names
    val_loss = 0

    visualizations = []
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            score = model(data)

        loss = cross_entropy2d(score, target, size_average=False)
        loss_data = loss.data.item()

        if np.isnan(loss_data):
            raise ValueError('loss is nan while validating')

        val_loss += loss_data / len(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = dataloader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)

            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=len(cls_names))
                visualizations.append(viz)

        metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds, len(cls_names))

        if len(visualizations) > 0:
            out = osp.join(out, 'visualization_viz')
            if not osp.exists(out):
                os.makedirs(out)
            out_file = osp.join(out, 'test.jpg')

            skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(dataloader)
