import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils import AverageMeter
# from tensorboardX import SummaryWriter

from data import get_vision_dataloader
from model import get_model

import sys
sys.path.append('systems')

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--system', type=str,
                        default='jetengine', help='Name of the dynamical system.')
parser.add_argument('--lambda', dest='_lambda', type=float, default=0.03, help='lambda for balancing the two loss terms.')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='Hyper-parameter in the hinge loss.')
parser.add_argument('--N_X0', type=int, default=100, help='Number of samples for the initial set X0.')
parser.add_argument('--N_x0', type=int, default=10, help='Number of samples for the initial state x0.')
parser.add_argument('--N_t', type=int, default=100, help='Number of samples for the time instant t.')
parser.add_argument('--layer1', type=int, default=64, help='Number of neurons in the first layer of the NN.')
parser.add_argument('--layer2', type=int, default=64, help='Number of neurons in the second layer of the NN.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--data_file_train', default='train.pklz', type=str, help='Path to the file for storing the generated training data set.')
parser.add_argument('--data_file_eval', default='eval.pklz', type=str, help='Path to the file for storing the generated evaluation data set.')
parser.add_argument('--log', type=str, help='Path to the directory for storing the logging files.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Use this option to disable cuda, if you want to train the NN on CPU.')
parser.set_defaults(use_cuda=True)

parser.add_argument('--bs', dest='batch_size', type=int, default=256)
parser.add_argument('--num_test', type=int, default=10)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


os.system('mkdir '+args.log)
os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), args.log))
os.system('cp *.py '+args.log)
os.system('cp -r systems/ '+args.log)
os.system('cp -r ODEs/ '+args.log)

config = importlib.import_module('system_'+args.system)
x, Ec, Er = config.sample_X0()
X0 = np.concatenate((x, Ec, Er))
model_r, forward_r = get_model(len(X0), 6, config, args)
if args.use_cuda:
    model_r = model_r.cuda()
else:
    model_r = model_r.cpu()

model_c, forward_c = get_model(len(X0), 6, config, args)
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

def hinge_loss_function(LHS, RHS):
    res = LHS - RHS + args.alpha
    res[res<0] = 0
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
        batch_size = x.size(0)
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if args.use_cuda:
            x = x.cuda()
            Ec = Ec.cuda()
            Er = Er.cuda()
            est = est.cuda()
            est_mean = est_mean.cuda()
        r = forward_r(torch.cat([x,Ec,Er], dim=1))
        c = forward_c(torch.cat([x,Ec,Er], dim=1))
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        loss_center = loss1(c, est_mean)
        DXi = xt - ref
        LHS = ((torch.matmul(TransformMatrix, DXi.view(batch_size,-1,1)).view(batch_size,-1)) ** 2).sum(dim=1)
        RHS = torch.ones(LHS.size()).type(DXi.type())

        _hinge_loss = hinge_loss_function(LHS, RHS)
        _volume_loss = -torch.log((TransformMatrix + 0.01 * torch.eye(TransformMatrix.shape[-1]).unsqueeze(0).type(X0.type())).det().abs())
        mask = _hinge_loss > 0
        _volume_loss[mask] = 0.
        _hinge_loss = _hinge_loss.mean()
        _volume_loss = _volume_loss.mean()

        CY2 = torch.sqrt(LHS)
        Y2 = torch.sqrt((DXi.view(batch_size,-1) ** 2).sum(dim=1))
        _l2_loss = (torch.abs((CY2 - 1)) * Y2 / CY2).mean()

        _loss = _hinge_loss + args._lambda * _volume_loss

        loss.update(_loss.item(), batch_size)
        prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss.item(), batch_size)
        volume_loss.update(_volume_loss.item(), batch_size)
        l2_loss.update(_l2_loss.item(), batch_size)

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
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        time_str += 'backward time: %.3f s'%(time.time()-tmp)
        end = time.time()

    print('Loss: %.3f, PREC: %.3f, HINGE_LOSS: %.3f, VOLUME_LOSS: %.3f, L2_loss: %.3f'%(loss.avg, prec.avg, hinge_loss.avg, volume_loss.avg, l2_loss.avg))

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
    adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    print('Epoch %d'%(epoch))
    _, _, _ = trainval(epoch, train_loader, writer=None, training=True)
    result_train, _, _ = trainval(epoch, train_loader, writer=None, training=False)
    result_val, loss, prec = trainval(epoch, val_loader, writer=None, training=False)
    epoch += 1
    # if prec > best_prec:
    if loss < best_loss:
        best_loss = loss
        # best_prec = prec
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()})