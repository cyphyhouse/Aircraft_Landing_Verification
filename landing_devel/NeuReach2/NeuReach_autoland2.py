import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils.utils import AverageMeter

from data import get_vision_dataloader
from model import get_model_rect
from datetime import datetime 
start_time = datetime.now()
start_time_str = start_time.strftime("%m-%d_%H-%M-%S")

import sys
sys.path.append('systems')

import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description="")
parser.add_argument('--system', type=str,
                        default='jetengine', help='Name of the dynamical system.')
parser.add_argument('--lambda', dest='_lambda', type=float, default=0.5, help='lambda for balancing the two loss terms.')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='Hyper-parameter in the hinge loss.')
parser.add_argument('--N_X0', type=int, default=100, help='Number of samples for the initial set X0.')
parser.add_argument('--N_x0', type=int, default=10, help='Number of samples for the initial state x0.')
parser.add_argument('--N_t', type=int, default=100, help='Number of samples for the time instant t.')
parser.add_argument('--layer1', type=int, default=64, help='Number of neurons in the first layer of the NN.')
parser.add_argument('--layer2', type=int, default=64, help='Number of neurons in the second layer of the NN.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training.')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--data_file_train', default='train.pklz', type=str, help='Path to the file for storing the generated training data set.')
parser.add_argument('--data_file_eval', default='eval.pklz', type=str, help='Path to the file for storing the generated evaluation data set.')
parser.add_argument('--log', type=str, default = os.path.join(script_dir, './log'),help='Path to the directory for storing the logging files.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Use this option to disable cuda, if you want to train the NN on CPU.')
parser.set_defaults(use_cuda=True)

parser.add_argument('--bs', dest='batch_size', type=int, default=256)
parser.add_argument('--num_test', type=int, default=10)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dim', '-d', type=str, default='x')

args = parser.parse_args()

args.data_file_train = os.path.join(script_dir, 'data.pickle')
args.data_file_eval = os.path.join(script_dir, 'data_eval.pickle')

np.random.seed(args.seed)
torch.manual_seed(args.seed)


os.system('mkdir '+args.log)
os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), args.log))
# os.system('cp *.py '+args.log)
# os.system('cp -r systems/ '+args.log)
# os.system('cp -r ODEs/ '+args.log)

# config = importlib.import_module('system_'+args.system)
import autoland_system as config
x, Ec, Er = config.sample_X0()
X0 = np.concatenate((x, Ec, Er))
if args.dim == 'x':
    model_r, forward_r = get_model_rect(3, 1, 64, 64)
    model_c, forward_c = get_model_rect(3, 1, 64, 64)
elif args.dim == 'y':
    model_r, forward_r = get_model_rect(4, 1, 64, 64)
    model_c, forward_c = get_model_rect(4, 1, 64, 64)

if args.use_cuda:
    model_r = model_r.cuda()
else:
    model_r = model_r.cpu()

if args.use_cuda:
    model_c = model_c.cuda()
else:
    model_c = model_c.cpu()


optimizer_r = torch.optim.Adam(model_r.parameters(), lr=args.learning_rate)
optimizer_c = torch.optim.Adam(model_c.parameters(), lr=args.learning_rate)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every * epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = args.log + '/' + filename
    torch.save(state, filename)

def hinge_loss_function(est, Radius, Center, alpha):
    tmp = torch.abs(Radius)
    if est.shape[1] == 21:
        res1 = torch.nn.ReLU()(est-(Center+tmp))
        res2 = torch.nn.ReLU()((Center-tmp)-est)
    else:
        est = torch.reshape(est, (est.shape[0], -1, 21))
        res1 = torch.nn.ReLU()(est-(Center+tmp).unsqueeze(2))
        res2 = torch.nn.ReLU()((Center-tmp).unsqueeze(2)-est)
    res = res1+res2
    return res

global_step = 0

def trainval(epoch, dataloader, writer, training):
    global global_step
    loss = AverageMeter()
    hinge_loss = AverageMeter()
    volume_loss = AverageMeter()
    l2_loss = AverageMeter()
    error_2 = AverageMeter()
    prec = AverageMeter()

    result = [[],[],[],[],[]] # for plotting

    if training:
        model_r.train()
        model_c.train()
    else:
        model_r.eval()
        model_c.train()
    end = time.time()
    loss1 = torch.nn.MSELoss()
    for step, (x, Ec, Er, est, est_mean) in enumerate(dataloader):
        batch_size = args.batch_size
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if args.use_cuda:
            x = x.cuda()
            Ec = Ec.cuda()
            Er = Er.cuda()
            est = est.cuda()
            est_mean = est_mean.cuda()
        data = torch.cat([x,Ec,Er], axis=1)
        Radius = forward_r(data)
        Center = forward_c(data)
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        _loss_total = 0
        _hinge_loss_total = 0
        _volume_loss_total = 0

        _hinge_loss = hinge_loss_function(est, Radius, Center, args.alpha)
        _volume_loss = torch.sum(torch.abs(Radius),1)
        _center_loss = torch.abs(Center - est_mean)

        _hinge_loss = _hinge_loss.mean()
        _volume_loss = _volume_loss.mean()
        _center_loss = _center_loss.mean()

        _loss = _center_loss
        # _loss = _center_loss + _hinge_loss + args._lambda*_volume_loss
        _loss_total += _loss
        _hinge_loss_total += _hinge_loss
        _volume_loss_total += _volume_loss

        # loss = _hinge_loss + args._lambda * _volume_loss + _center_loss

        loss.update(_loss.item(), batch_size)
        # prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss.item(), batch_size)
        volume_loss.update(_volume_loss.item(), batch_size)
        # l2_loss.update(_l2_loss.item(), batch_size)
        print(epoch, step, round(_loss.item(),2), round(_hinge_loss.item(),2), round(_volume_loss.item(),2), round(_center_loss.item(),2))

        if writer is not None and training:
            writer.add_scalar('loss', loss.val, global_step)
            writer.add_scalar('prec', prec.val, global_step)
            writer.add_scalar('Volume_loss', volume_loss.val, global_step)
            writer.add_scalar('Hinge_loss', hinge_loss.val, global_step)
            writer.add_scalar('L2_loss', l2_loss.val, global_step)

        time_str += 'other time: %.3f s\t'%(time.time()-end)
        tmp = time.time()
        if training:
            global_step += 1
            # optimizer_r.zero_grad()
            optimizer_c.zero_grad()
            _loss.backward()
            # optimizer_r.step()
            optimizer_c.step()
        time_str += 'backward time: %.3f s'%(time.time()-tmp)
        end = time.time()

    # print('Loss: %.3f, PREC: %.3f, HINGE_LOSS: %.3f, VOLUME_LOSS: %.3f, L2_loss: %.3f'%(loss.avg, prec.avg, hinge_loss.avg, volume_loss.avg, l2_loss.avg))

    if writer is not None and not training:
        writer.add_scalar('loss', loss.avg, global_step)
        writer.add_scalar('prec', prec.avg, global_step)
        writer.add_scalar('Volume_loss', volume_loss.avg, global_step)
        writer.add_scalar('Hinge_loss', hinge_loss.avg, global_step)
        writer.add_scalar('L2_loss', l2_loss.avg, global_step)

    return result, loss.avg, prec.avg

train_loader, val_loader = get_vision_dataloader(config, args)

# train_writer = SummaryWriter(args.log+'/train')
# val_writer = SummaryWriter(args.log+'/val')

best_loss = np.inf
best_prec = 0

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer_r, epoch)
    adjust_learning_rate(optimizer_c, epoch)
    # train for one epoch
    print('Epoch %d'%(epoch))
    _, _, _ = trainval(epoch, train_loader, writer=None, training=True)
    result_train, loss, _ = trainval(epoch, train_loader, writer=None, training=False)
    result_val, loss, prec = trainval(epoch, val_loader, writer=None, training=False)
    epoch += 1
    # if prec > best_prec:
    if loss < best_loss:
        best_loss = loss
        print(best_loss)
        # best_prec = prec
        # save_checkpoint({'epoch': epoch + 1, 'state_dict': model_r.state_dict()}, filename=f"checkpoint_{args.dim}_r_{start_time_str}_{epoch}.pth.tar")
        # save_checkpoint({'epoch': epoch + 1, 'state_dict': model_c.state_dict()}, filename=f"checkpoint_{args.dim}_c_{start_time_str}_{epoch}.pth.tar")
