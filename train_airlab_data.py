#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml
from torch.utils.data import DataLoader

import torchfcn
from cmu_airlab.datasets.dataset_air_lab import AirLabClassSegBase
from train_fcn32s import get_parameters, git_hash

here = osp.dirname(osp.abspath(__file__))


def main():
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
    args = parser.parse_args()

    args.model = 'FCN8s'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/Daten/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = DataLoader(AirLabClassSegBase(root, transform=True), batch_size=1, shuffle=False, **kwargs)
    val_loader = DataLoader(AirLabClassSegBase(root, val=True, transform=True), batch_size=1, shuffle=False, **kwargs)

    # 2. model

    fcn_model = torchfcn.models.FCN8s(n_class=11)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        fcn_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn16s = torchfcn.models.FCN16s()
        state_dict = torch.load(args.pretrained_model)
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        fcn_model.copy_params_from_fcn16s(fcn16s, n_class_changed=True)
    if cuda:
        print("Using CUDA.")
        fcn_model = fcn_model.cuda()

    ct = 0

    # Freezing cnn weights.
    for name, layer in fcn_model.named_children():
        if name not in fcn_model.class_dependent_layers:
            for param in layer.parameters():
                param.requires_grad = False

    # 3. optimizer

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
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=fcn_model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=50,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
