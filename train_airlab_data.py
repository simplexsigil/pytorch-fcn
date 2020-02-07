#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import os
import os.path as osp
import uuid

import torch
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchfcn
from cmu_airlab.datasets.dataset_air_lab import AirLabClassSegBase
from torchfcn.models.fcn_utils import get_parameters
from torchfcn.utils import git_hash

# This is used to differentiate a kind of 'debug' mode on my notebook, which does not have enough graphics memory.
nb_hash = b'\x88\x95\xe23\x9b\xff_RN8\xfe\xd0\x08\xe6r\x05m1\x9e\x94\xac!\xef\xb2\xc2\xc9k\x18\x0f\xc6\xda\xbf'
here = osp.dirname(osp.abspath(__file__))


def main():
    m = hashlib.sha256()
    m.update(str(uuid.getnode()).encode('utf-8'))
    on_my_notebook = m.digest() == nb_hash

    args = argument_parsing()

    args.model = 'FCN8s'
    args.git_hash = git_hash()  # This is a nice idea: Makes results reproducible by logging current git commit.

    args.use_cuda = prepare_cuda(args, torch_seed=42)
    args.use_cuda = False if on_my_notebook else args.use_cuda

    settings_to_logfile(args)

    print("Output folder:\n{}".format(args.out))

    for k in range(args.k_fold):
        print("Training fold {}/{}".format(k, args.k_fold))

        out = osp.join(args.out, "fold_{}".format(k))
        # Prepare Dataset
        root = osp.expanduser('~/Daten/datasets')
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}

        train_dst = AirLabClassSegBase(root, transform=True, max_len=3 if on_my_notebook else None,
                                       k_fold=args.k_fold, k_fold_val=k, use_augmented=True)

        test_dst = AirLabClassSegBase(root, val=True, transform=True, max_len=3 if on_my_notebook else None,
                                      k_fold=args.k_fold, k_fold_val=k, use_augmented=False)

        train_loader = DataLoader(train_dst, batch_size=5, shuffle=False, **kwargs)
        val_loader = DataLoader(test_dst, batch_size=1, shuffle=False, **kwargs)

        # Check for checkpoint.
        start_epoch = 0
        start_iteration = 0
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            start_iteration = checkpoint['iteration']

        # Prepare model. Load weights from checkpoint if available.
        fcn_model = prepare_model(args, freeze_cnn_weights=True, checkpoint=checkpoint)

        # Prepare optimizer and learning rate scheduler-
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(fcn_model, bias=False)},
                {'params': get_parameters(fcn_model, bias=True),
                 'lr': args.lr * 2, 'weight_decay': 0},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        if args.resume:
            checkpoint = torch.load(args.resume)
            optim.load_state_dict(checkpoint['optim_state_dict'])

        scheduler = MultiStepLR(optim, milestones=[70, 80, 90], gamma=0.1, last_epoch=start_epoch - 1)

        weight_unfreezer = prepare_weight_unfreezer(optim, fcn_model, cnn_weights_frozen=True)
        model_refiner = prepare_model_refinement(fcn_model)

        trainer = torchfcn.Trainer(
            cuda=args.use_cuda,
            model=fcn_model,
            optimizer=optim,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            out=out,
            max_epoch=args.max_epoch,
            interval_val_viz=5,
            epoch_callback_tuples=[(30, model_refiner), (70, weight_unfreezer)]
        )

        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()


def prepare_model_refinement(fcn_model):
    def set_model_refinement():
        fcn_model.use_refinement = True

        print("Model is using refinement layer, now.")

    return set_model_refinement


def prepare_weight_unfreezer(optim, fcn_model, cnn_weights_frozen):
    def weight_unfreezer():
        if cnn_weights_frozen:  # Freezing cnn weights.
            for name, layer in fcn_model.named_children():
                if name not in fcn_model.class_dependent_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

        print("CNN weights unfrozen.")

    return weight_unfreezer


def prepare_model(args, freeze_cnn_weights=True, checkpoint=None):
    fcn_model = torchfcn.models.FCN8s(n_class=11)

    if checkpoint is not None:
        fcn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # It seem to be tedious to load the pretrained model into FCN16s first and then copy the params from there.
        # I assume this is due to the available pretrained models.
        fcn16s = torchfcn.models.FCN16s()
        state_dict = torch.load(args.pretrained_model)
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        fcn_model.copy_params_from_fcn16s(fcn16s, n_class_changed=True)
    if args.use_cuda:
        print("Using CUDA.")
        fcn_model = fcn_model.cuda()

    if freeze_cnn_weights:  # Freezing cnn weights.
        for name, layer in fcn_model.named_children():
            if name not in fcn_model.class_dependent_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    return fcn_model


def settings_to_logfile(args):
    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


def prepare_cuda(args, torch_seed=42):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.cuda.manual_seed(torch_seed)
    return use_cuda


def argument_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu img_id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-epoch', type=int, default=101, help='max epoch'
    )
    parser.add_argument(
        '--k-fold', type=int, default=4, help='k for k-fold validation'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-8, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--pretrained-model',
        default=torchfcn.models.FCN16s.download(),
        help='pretrained model of FCN16s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
