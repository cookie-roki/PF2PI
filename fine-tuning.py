# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
# --------------------------------------------------------

import argparse
import os
import random
import datetime
import time
from typing import List
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from get_dataset import get_dataset
import aslloss
from predictor import build_predictor
from validation import evaluate_performance
from logger import setup_logger
import csv


def parser_args():
    parser = argparse.ArgumentParser(description='CFAGO main')
    parser.add_argument('--org', help='organism')
    parser.add_argument('--dataset_dir', help='dir of dataset')
    parser.add_argument('--aspect', type=str, choices=['P', 'F', 'C'], help='GO aspect')
    parser.add_argument('--pretrained_model', type=str, help='pretrained self-supervide learning model')

    parser.add_argument('--output', metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--num_class', default=45, type=int,
                        help="Number of class labels")
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--norm_norm', action='store_true', default=False,
                        help='using mormal scale to normalize input features')

    # * Transformer
    parser.add_argument('--attention_layers', default=6, type=int,
                        help="Number of layers of each multi-head attention module")

    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the multi-head attention blocks")
    parser.add_argument('--activation', default='gelu', type=str, choices=['relu', 'gelu', 'lrelu', 'sigmoid'],
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the multi-head attention module")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * raining
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


best_mAP = 0


def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = get_args()

    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        set_rand_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # torch.distributed.init_process_group(backend='gloo', init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="CFAGO")

    args.h_n = 1
    return main_worker(args, logger)


def main_worker(args, logger):
    global best_mAP
    train_dataset, test_dataset, args.modesfeature_len = get_dataset(args)
    criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 32

    # tensorboard

    # optionally resume from a checkpoint

    # Data loading code
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = None
    # assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, losses_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    # pre_model_scheduler = lr_scheduler.OneCycleLR(pre_model_optimizer, max_lr=args.lr, steps_per_epoch=full_dataset[0].shape[0]//args.batch_size, epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_finetune_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_regular_finetune_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()

    fn = args.output + '/' + args.org + '_attention_layers_' + str(
        args.attention_layers) + '_aspect_' + args.aspect + '_fintune_seed_' + str(args.seed) + \
         '_act_' + args.activation + '.csv'
    with open(fn, 'w') as f:
        csv.writer(f).writerow(['m-aupr', 'M-aupr', 'F1', 'acc', 'Fmax'])

    for epoch in range(1):
        if args.seed is not None:
            set_rand_seed(args.seed)
        torch.cuda.empty_cache()

        finetune_pre_model = torch.load(args.pretrained_model)
        predictor_model = build_predictor(finetune_pre_model, args)
        predictor_model = predictor_model.cuda()

        # if args.optim == 'AdamW':
        predictor_model_param_dicts = [
            {"params": [p for n, p in predictor_model.pre_model.named_parameters() if p.requires_grad], "lr": 1e-5},
            {"params": [p for n, p in predictor_model.fc_decoder.named_parameters() if p.requires_grad]}
        ]

        predictor_model_optimizer = getattr(torch.optim, 'AdamW')(
            predictor_model_param_dicts,
            lr=args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
        steplr = lr_scheduler.StepLR(predictor_model_optimizer, 50)
        patience = 10
        changed_lr = False
        for epoch_train in range(100):
            # train for one epoch
            train_loss = train(train_loader, predictor_model, criterion,
                               predictor_model_optimizer, steplr, epoch_train, args, logger)
            print('loss = ', train_loss)

            old_loss = train_loss

        # evaluate on testing set
        loss, perf = evaluate(test_loader, predictor_model, criterion, args, logger)

        with open(fn, 'a') as f:
            csv.writer(f).writerow([perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc'], perf['Fmax']])

    return 0


def train(train_loader, predictor_model, criterion, optimizer, steplr, epoch_train, args, logger):
    losses = AverageMeter('Loss', ':5.3f')
    # lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    def get_learning_rate(optimizer):
        return optimizer.param_groups[1]["lr"]

    # lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    predictor_model.train()
    # switch to train mode

    if epoch_train >= 50:
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = True
    else:
        # finetune_pre_model.eval()
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = False

    end = time.time()
    for i, (proteins, label) in enumerate(train_loader):
        # measure data loading time
        proteins[0] = proteins[0].cuda()
        proteins[1] = proteins[1].cuda()
        label = label.cuda()
        # compute output

        rec, output = predictor_model(proteins)
        loss = criterion(rec, output, label)
        if args.loss_dev > 0:
            loss *= args.loss_dev

        # record loss
        losses.update(loss.item(), proteins[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    steplr.step()
    return losses.avg


@torch.no_grad()
def evaluate(test_loader, predictor_model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    # Acc1 = AverageMeter('Acc@1', ':5.2f')
    # top5 = AverageMeter('Acc@5', ':5.2f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    # mAP = AverageMeter('mAP', ':5.3f', val_only=)

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    predictor_model.eval()
    saved_data = []
    with torch.no_grad():
        for i, (proteins, label) in enumerate(test_loader):
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            label = label.cuda()

            # compute output
            rec, output = predictor_model(proteins)
            loss = criterion(rec, output, label)
            if args.loss_dev > 0:
                loss *= args.loss_dev
            output_sm = nn.functional.sigmoid(output)
            # output_sm = output
            if torch.isnan(loss):
                saveflag = True

            # record loss
            losses.update(loss.item(), proteins[0].size(0))

            # save some data
            # output_sm = nn.functional.sigmoid(output)
            _item = torch.cat((output_sm.detach().cpu(), label.detach().cpu()), 1)
            # del output_sm
            # del target
            saved_data.append(_item)

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        # logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )
        print('saved_data shape = ', len(saved_data), ', ', saved_data[0].shape)
        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            # filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = evaluate_performance
            # mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            y_score = saved_data[:, 0:(saved_data.shape[1] // 2)]
            labels_val = saved_data[:, (saved_data.shape[1] // 2):]
            perf = metric_func(labels_val, y_score, (y_score > 0.5).astype(int))
            print('%0.5f %0.5f %0.5f %0.5f %0.5f\n' % (
            perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc'], perf['Fmax']))

            # logger.info(" m_aupr: {}, M_aupr: {}, fmax: {}, acc: {}, ".format(perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc']))
        else:
            perf = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, perf


##################################################################################
def add_weight_decay(pre_model, decoder_modal, weight_decay=1e-4, skip_list=()):
    pre_decay = []
    pre_no_decay = []
    decode_decay = []
    decode_no_decay = []
    for name, param in pre_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            pre_no_decay.append(param)
        else:
            pre_decay.append(param)
    for name, param in decoder_modal.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            decode_no_decay.append(param)
        else:
            decode_decay.append(param)
    return [
        {'params': pre_no_decay, 'weight_decay': 0.},
        {'params': pre_decay, 'weight_decay': weight_decay},
        {'params': decode_no_decay, 'weight_decay': 0.},
        {'params': decode_decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class myRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data, generator, shuffle=True):
        self.data = data
        self.generator = generator
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            return iter(torch.randperm(n, generator=self.generator).tolist())
        else:
            return iter(list(range(n)))

    def __len__(self):
        return len(self.data)


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


if __name__ == '__main__':
    main()
