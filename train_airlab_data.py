#!/usr/bin/env python3

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchfcn
from cmu_airlab.datasets.dataset_air_lab import AirLabClassSegBase
from torchfcn.models.fcn_utils import get_parameters
from torchfcn.utils import git_hash

here = osp.dirname(osp.abspath(__file__))


def main():
    args = argument_parsing()

    args.model = 'FCN8s'
    args.git_hash = git_hash()  # This is a nice idea: Makes results reproducible by logging current git commit.

    args.use_cuda = prepare_cuda(args, torch_seed=42)
    args.use_cuda = False

    settings_to_logfile(args)

    # Prepare Dataset
    root = osp.expanduser('~/Daten/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.use_cuda else {}

    train_loader = DataLoader(AirLabClassSegBase(root, transform=True), batch_size=1, shuffle=False, **kwargs)
    val_loader = DataLoader(AirLabClassSegBase(root, val=True, transform=True), batch_size=1, shuffle=False, **kwargs)

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

    scheduler = MultiStepLR(optim, milestones=[15, 40, 80, 140, 200], gamma=0.1, last_epoch=start_epoch - 1)

    weight_unfreezer = prepare_weight_unfreezer(optim, fcn_model, cnn_weights_frozen=True)

    trainer = torchfcn.Trainer(
        cuda=args.use_cuda,
        model=fcn_model,
        optimizer=optim,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_epoch=args.max_iteration,
        interval_val_viz=5,
        epoch_callback_tuples=[(30, weight_unfreezer)]
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


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
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-14, help='learning rate',
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
