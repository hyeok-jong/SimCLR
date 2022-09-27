from __future__ import print_function
import argparse
import time

import datetime

import torch
import torch.backends.cudnn as cudnn

import wandb
from util import init_wandb
from torchvision import transforms, datasets
from util import AverageMeter, accuracy
from util import adjust_learning_rate, warmup_learning_rate, get_learning_rate
from util import set_optimizer

from models import ResNetSimCLR, VGGSimCLR, AlexSimCLR

def feature_distance_parser():
    parser = argparse.ArgumentParser('argument for LPIPS distance evaluation')
    parser.add_argument('-d','--distortion', type = str, default = None,
                        help = '[supcon rand v1]')
    parser.add_argument('--dataset', type = str, default = 'cifar10',
                        help = '[BAPPS / cifar10 / cifar100 / stl10]')
    parser.add_argument('--test_dataset', type = str, default = 'cifar10',
                        help = '[cifar10 / cifar100]')
    parser.add_argument('--model', type = str, default = 'resnet18')
    parser.add_argument('--batch_size', type = int, default = 512)
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--size', type = int, default = 64)
    parser.add_argument('--test_epoch', type = int, default = 100)
    parser.add_argument('--epochs', type = int, default = 200)
    # wandb
    parser.add_argument('--wandb_entity', type=str, default='hyeokjong',
                        help='Wandb ID')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Project name')
    parser.add_argument('--short', type=str, default=None,
                        help='short name')

    # Optimization
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--learning_rate', type = float, default = 0.5,
                        help = 'learning rate')
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')
    parser.add_argument('--cosine', action = 'store_true',
                        help = 'using cosine annealing')
    parser.add_argument('--warm', action = 'store_true',
                        help = 'warm-up for large batch training')
    parser.add_argument('--warmup-from', type=float, default=0.01)
    parser.add_argument('--warm-epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60, 80, 90],
                        help='where to decay lr, can be a list')                        
    args = parser.parse_args()

    args.path = f'./result/{args.distortion}/{args.dataset}/{args.model}/{args.size}_{args.test_epoch}.pt'
    
    if args.wandb_project == None:
        args.wandb_project = f'[SimCLR][{args.distortion}][{args.dataset}][{args.model}][input{args.size}]'
    if args.short == None:
        args.short = f'[Linear Evaluation][test dataset : {args.test_dataset}][batch:{args.batch_size}][trained epoch:{args.test_epoch}][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'

    return args


def set_loader(args):
    # construct data loader
    if args.test_dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    elif args.test_dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=args.size),
        transforms.ToTensor(),
        normalize,
    ])

    if args.test_dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./datasets',
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root='./datasets',
                                       train=False,
                                       transform=val_transform)

    elif args.test_dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./datasets',
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root='./datasets',
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(args.test_dataset)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=6, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader



def set_model(args):

    if args.test_dataset == 'cifar10':
        num_classes = 10
    elif args.test_dataset == 'cifar100':
        num_classes = 100

    if args.model[:3] == 'res':
        model = ResNetSimCLR(base_model = args.model, mode = 'Linear', out_dim = num_classes).to(args.device)

    elif args.model[:3] == 'vgg':
        model = VGGSimCLR(base_model = args.model, mode = 'Linear', out_dim = num_classes).to(args.device)

    elif args.model[:4] == 'alex':
        model = AlexSimCLR(base_model = args.model, mode = 'Linear', out_dim = num_classes).to(args.device)


    # Load pre-trained encoder
    state_dict = torch.load(args.path)['state_dict']
    print(model.load_state_dict(state_dict, strict = False))

    # Freeze encoder
    for name, param in model.named_parameters():
        if name[:4] != 'head':
            param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    cudnn.benchmark = True
    print(model)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""

    model = model.to(args.device)
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):


        images = images.to(args.device)
        labels = labels.to(args.device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # Forward
        output = model(images)

        # Loss
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res = {
        'training_loss': losses.avg,
        'learning_rate': get_learning_rate(optimizer),
        'training_top1_acc': top1.avg,
        'training_top5_acc': top5.avg
            }
    return res


def validate(val_loader, model, criterion, args):

    model = model.to(args.device)
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(args.device)
            labels = labels.to(args.device)
            bsz = labels.shape[0]

            # Forward
            output = model(images)

            # Loss
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            res = {
            'validation_loss': losses.avg,
            'validation_top1_acc': top1.avg,
            'validation_top5_acc': top5.avg
                }
    return res


def main():
    best_acc = 0
    args = feature_distance_parser()
    init_wandb(args)

    # build data loader
    train_loader, val_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    print(model)
    for i in model.parameters():
        print(i.requires_grad)

    # build optimizer
    optimizer = set_optimizer(args, model)

    # training routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        res_train = train(train_loader, model, criterion,
                          optimizer, epoch, args)
        time2 = time.time()

        # eval for one epoch
        res_val = validate(val_loader, model, criterion, args)
        if res_val['validation_top1_acc'] > best_acc:
            best_acc = res_val['validation_top1_acc']

        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, res_train['training_top1_acc'], res_val['validation_top1_acc']))

        wandb.log(res_train  | res_val, step = epoch)

    wandb.finish()

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
