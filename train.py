from __future__ import print_function

import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import datetime
import wandb

from util import AverageMeter
from util import format_time, set_dir
from util import adjust_learning_rate, warmup_learning_rate, get_learning_rate
from util import set_optimizer, init_wandb, save_logs
from losses import SupConLoss
from parser import SupCon_parser

from models import ResNetSimCLR, VGGSimCLR, AlexSimCLR
from data import set_loader

def set_model(args):

    if args.model[:3] == 'res':
        model = ResNetSimCLR(base_model = args.model, out_dim = args.out_dim).to(args.device)

    elif args.model[:3] == 'vgg':
        model = VGGSimCLR(base_model = args.model, out_dim = args.out_dim).to(args.device)

    elif args.model[:4] == 'alex':
        model = AlexSimCLR(base_model = args.model, out_dim = args.out_dim).to(args.device)

    criterion = SupConLoss(temperature=args.temp, device = args.device).to(args.device)

    cudnn.benchmark = True
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.train()

    losses = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
    
        bsz = images[0].shape[0]
        # labels = labels.to(args.device)
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(args.device)
        
        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # Forward
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Loss
        if args.method == 'SupCon':
            loss = criterion(features, labels)
        elif args.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(args.method))

        # Update loss
        losses.update(loss.item(), bsz)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res = {
        'train_loss': losses.avg,
        'learning_rate': get_learning_rate(optimizer)
            }
    return res




def main():
    set_dir()
    args = SupCon_parser()

    init_wandb(args)

    # build data loader
    train_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)


    # training routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        start_time = time.time()
        res = train(train_loader, model, criterion, optimizer, epoch, args)
        loss = res['train_loss']
        lr = res['learning_rate']
        print(f'[epoch:{epoch}/{args.epochs}] [loss:{loss}] [lr:{np.round(lr,6)}] [Time:[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]] [total time:{format_time(time.time() - start_time)}]')

        wandb.log(res, step = epoch)

        # Custom Save logs
        save_logs(args, epoch, model, optimizer, loss, lr)

    wandb.finish()

if __name__ == '__main__':
    main()
