'''
Training wiht VLOG
'''
from __future__ import print_function

import sys


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"


from comet_ml import Experiment as CometExperiment, OfflineExperiment

import argparse
import os
import shutil
import time
import random
import cv2

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import utils.imutils2
import models.videos.model_simple as models
from utils import Logger, AverageMeter, mkdir_p, savefig

import models.dataset.vlog_train as vlog

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j',
                    '--workers',
                    default=12,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=2e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.5,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0.0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument(
    '-c',
    '--checkpoint',
    default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/',
    type=str,
    metavar='PATH',
    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id',
                    default='0,1,2,3',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--predDistance',
                    default=4,
                    type=int,
                    help='predict how many frames away')
parser.add_argument('--seperate2d', type=int, default=0, help='manual seed')
parser.add_argument('--batchSize', default=36, type=int, help='batchSize')
parser.add_argument('--T', default=512**-.5, type=float, help='temperature')
parser.add_argument('--gridSize', default=9, type=int, help='temperature')
parser.add_argument('--classNum', default=49, type=int, help='temperature')
parser.add_argument('--lamda', default=0.1, type=float, help='temperature')
parser.add_argument('--pretrained_imagenet',
                    type=str_to_bool,
                    nargs='?',
                    const=True,
                    default=False,
                    help='pretrained_imagenet')

parser.add_argument('--videoLen', default=4, type=int, help='')
parser.add_argument('--frame_gap', default=2, type=int, help='')

parser.add_argument('--hist', default=1, type=int, help='')
parser.add_argument('--optim', default='adam', type=str, help='')

parser.add_argument('--no_log_to_comet', action='store_true', default=False)
parser.add_argument('--log_to_comet_every', default=50, type=int)
parser.add_argument('--log_image_every', default=100, type=int)
parser.add_argument(
    '--dataset',
    default='vlog',
    type=str,
    help=
    'types are vlog, those defined in data/gymnastics.py. can combine vlog and gymnastics by doing e.g. `vlog.jun1full-gymnastics`'
)
parser.add_argument(
    '--vlog_train',
    default=
    '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/VLOG/vlog_train.txt',
    type=str)
parser.add_argument(
    '--vlog_test',
    default=
    '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/VLOG/vlog_test.txt',
    type=str)
parser.add_argument(
    '--vlog_val',
    default=
    '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/VLOG/vlog_val.txt',
    type=str)
parser.add_argument(
    '--gymnastics_dataset_location',
    type=str,
    default='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/jun-01-2019',
    help="the location of the dataset. this will differ based on cluster.")
parser.add_argument('--imgSize', default=256, type=int, help='')
parser.add_argument('--imgSize2', default=320, type=int, help='')
parser.add_argument('--cropSize', default=240, type=int, help='')
parser.add_argument('--cropSize2', default=80, type=int, help='')
parser.add_argument('--offset', default=0, type=int, help='')
parser.add_argument('--local_comet_dir',
			   type=str,
			   default=None,
			   help='local dir to process comet locally only. '
			   'primarily for fb, will stop remote calls.')
parser.add_argument('--name',
			   type=str,
			   help='the identifying name of this experiment.',
			   default=None)
parser.add_argument('--counter',
			   type=int,
			   help='the integer counter of this experiment. '
			   'defaults to None because Cinjon is likely the '
			   'only one who is going to use it.')

        
args = parser.parse_args()
params = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_loss = 0  # best test accuracy

if not args.no_log_to_comet:
    if params['local_comet_dir']:
        comet_exp = OfflineExperiment(
            api_key="hIXq6lDzWzz24zgKv7RYz6blo",
            project_name="supercyclecons",
            workspace="cinjon",
            auto_metric_logging=True,
            auto_output_logging=None,
            auto_param_logging=False,
            offline_directory=params['local_comet_dir'])
    else:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="supercyclecons",
                                    workspace="cinjon",
                                    auto_metric_logging=True,
                                    auto_output_logging=None,
                                    auto_param_logging=False)
    comet_exp.log_parameters(vars(args))
    comet_exp.set_name(params['name'])


def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict
    }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)


def main():
    global best_loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_global_step = 0

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.CycleTime(class_num=params['classNum'],
                             trans_param_num=3,
                             pretrained=args.pretrained_imagenet,
                             temporal_out=params['videoLen'],
                             T=args.T,
                             hist=args.hist)

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss().cuda()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=(args.momentum, 0.999),
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=0.95)

    if len(args.pretrained) > 0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.pretrained), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrained)

        partial_load(checkpoint['state_dict'], model)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

        del checkpoint

    title = 'videonet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_global_step = checkpoint['global_step']

        partial_load(checkpoint['state_dict'], model)

        logger = Logger(os.path.join(args.checkpoint, 'log-resume.txt'),
                        title=title)
        logger.set_names(
            ['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])

        del checkpoint

    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(
            ['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])

    dataset = vlog.VlogSet(params, is_train=True, frame_gap=args.frame_gap)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params['batchSize'],
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    start_time = time.time()
    
    # Train and val
    global_step = start_global_step
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, args.epochs, params['lr']))

        train_loss, theta_loss, theta_skip_loss, global_step = train(
            train_loader, model, criterion, optimizer, epoch, global_step,
            start_global_step, start_time, use_cuda, args)
            
        if not args.no_log_to_comet:
            comet_exp.log_epoch_end(epoch)

        # append logger file
        logger.append([params['lr'], train_loss, theta_loss, theta_skip_loss])

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': model.state_dict(),
            },
            checkpoint=args.checkpoint
        )

    logger.close()


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def _get_box(_theta, _img, _color):
    crop_size = params['cropSize']
    flow = F.affine_grid(_theta, torch.Size((1, 3, crop_size, crop_size)))
    points = np.zeros((4, 2), dtype=int)
    points[0] = ((flow[0, 0, 0] + 1) * crop_size / 2).detach().cpu().numpy()
    points[1] = ((flow[0, -1, 0] + 1) * crop_size / 2).detach().cpu().numpy()
    points[2] = ((flow[0, -1, -1] + 1) * crop_size / 2).detach().cpu().numpy()
    points[3] = ((flow[0, 0, -1] + 1) * crop_size / 2).detach().cpu().numpy()

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box_d = np.int0(box)
    cv2.drawContours(_img, [box_d], 0, _color, 2)

    return _img


def train(train_loader, model, criterion, optimizer, epoch, global_step, start_global_step, start_time, use_cuda, args):
    if not args.no_log_to_comet:
        comet_exp.log_current_epoch(epoch)
        comet_exp.log_metric('epoch', epoch, step=global_step)        

    # switch to train mode
    model.train()
    # model.apply(set_bn_eval)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    main_loss = AverageMeter()
    losses_theta = AverageMeter()
    losses_theta_skip = AverageMeter()

    losses_dict = dict(cnt_trackers=None,
                       back_inliers=None,
                       loss_targ_theta=None,
                       loss_targ_theta_skip=None,
                       loss_back_inliers=None
    )

    end = time.time()


    for batch_idx, (imgs, img, patch2, theta, meta) in enumerate(train_loader):
        global_step += 1
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        # optimizerC.zero_grad()

        if imgs.size(0) < params['batchSize']:
            break

        imgs = torch.autograd.Variable(imgs.cuda())
        img = torch.autograd.Variable(img.cuda())
        patch2 = torch.autograd.Variable(patch2.cuda())
        theta = torch.autograd.Variable(theta.cuda())

        folder_paths = meta['folder_path']
        startframes = meta['startframe']
        future_idxs = meta['future_idx']

        outputs = model(imgs, patch2, img, theta)

        losses = model.module.loss(*outputs)
        loss_targ_theta, loss_targ_theta_skip, loss_back_inliers = losses

        # adjusting coefficient for stable training
        loss = sum(loss_targ_theta) / len(loss_targ_theta) * args.lamda * 0.2 + \
         sum(loss_back_inliers) / len(loss_back_inliers) + \
         loss_targ_theta_skip[0] * args.lamda

        outstr = ''

        main_loss.update(loss_back_inliers[0].data, imgs.size(0))
        outstr += '| Loss: %.3f' % (main_loss.avg)

        losses_theta.update(
            sum(loss_targ_theta).data / len(loss_targ_theta), imgs.size(0))
        losses_theta_skip.update(
            sum(loss_targ_theta_skip).data / len(loss_targ_theta_skip),
            imgs.size(0))

        def add_loss_to_str(name, _loss):
            outstr = ' | %s ' % name
            if losses_dict[name] is None:
                losses_dict[name] = [AverageMeter() for _ in _loss]

            for i, l in enumerate(_loss):
                losses_dict[name][i].update(l.data, imgs.size(0))
                outstr += ' %s: %.3f ' % (i, losses_dict[name][i].avg)
            return outstr

        outstr += add_loss_to_str('loss_targ_theta', loss_targ_theta)
        outstr += add_loss_to_str('loss_targ_theta_skip', loss_targ_theta_skip)
        outstr += add_loss_to_str('loss_back_inliers', loss_back_inliers)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                
        secs = time.time() - start_time
        steps_per_sec = (global_step - start_global_step) * 1.0 / secs
        
        if batch_idx % 5 == 0:
            outstr = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | S/S: {sps:.3f} | {outstr}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                data=data_time.val,
                bt=batch_time.val,
                sps=steps_per_sec,
                outstr=outstr)
            print(outstr)

        if not args.no_log_to_comet and global_step % args.log_to_comet_every == 0:
            comet_exp.log_metric('steps_per_sec', steps_per_sec, global_step)
            
            with comet_exp.train():
                avg_loss_targ_theta = sum(loss_targ_theta).data / len(loss_targ_theta)
                avg_loss_targ_theta *= args.lamda
                avg_loss_targ_theta *= 0.2

                avg_loss_back_inliers = sum(loss_back_inliers).data / len(loss_back_inliers)

                avg_loss_targ_theta_skip = sum(loss_targ_theta_skip).data / len(
                    loss_targ_theta_skip)
                avg_loss_targ_theta_skip *= args.lamda
                
                metrics = {
                    'avg_loss_targ_theta': avg_loss_targ_theta,
                    'avg_loss_back_inliers': avg_loss_back_inliers,
                    'avg_loss_targ_theta_skip': avg_loss_targ_theta_skip,
                    'loss_': avg_loss_targ_theta + avg_loss_back_inliers + avg_loss_targ_theta_skip,
                    'loss': loss
                }
                comet_exp.log_metrics(metrics, step=global_step)

        if not args.no_log_to_comet and global_step % args.log_image_every == 0:
            with comet_exp.train():
                # Image logging
                forw_trans_thetas = torch.stack(outputs[0][1])
                if global_step % args.log_image_every == 0:
                    log_img = np.transpose(
                        img[0, 0].detach().cpu().numpy().copy(), (1, 2, 0))
                    for c in range(3):
                        log_img[:, :, c] = log_img[:, :, c] * std[c]
                        log_img[:, :, c] = log_img[:, :, c] + mean[c]
                    log_img = np.array(log_img.copy())
                    log_img = _get_box(theta[:1], log_img, (0., 0., 1.))
                    log_img = _get_box(forw_trans_thetas[-1, :1], log_img,
                                       (1., 0., 0.))

                    img_name = 'epch{}_gs{}'.format(epoch, global_step)
                    comet_exp.log_image(log_img,
                                        name=img_name,
                                        image_channels='last',
                                        copy_to_tmp=False)

    return main_loss.avg, losses_theta.avg, losses_theta_skip.avg, global_step


def save_checkpoint(state,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
