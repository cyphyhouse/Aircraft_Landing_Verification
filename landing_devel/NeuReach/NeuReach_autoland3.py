import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from utils import AverageMeter

from datetime import datetime 
start_time = datetime.now()
start_time_str = start_time.strftime("%m-%d_%H-%M-%S")

from data import get_dataloader_autoland2
from model import get_model_rect2, get_model_rect

import sys
sys.path.append('systems')

import os 
import argparse

global_step = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every * epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = args.log + '/' + filename
    torch.save(state, filename)

def hinge_loss_function(LHS, RHS, alpha):
    res1 = LHS - RHS[:,0] + alpha
    res1 = (torch.nn.ReLU())(res1)
    res2 = -RHS[:,1] - LHS + alpha 
    res2 = (torch.nn.ReLU())(res2)
    res = res1 + res2 
    res = res.sum(dim=1)
    return res

def trainval(model, forward, optimizer, epoch, dataloader, writer, training, alpha, _lambda):
    global global_step
    loss = AverageMeter()
    hinge_loss = AverageMeter()
    volume_loss = AverageMeter()
    l2_loss = AverageMeter()
    error_2 = AverageMeter()
    prec = AverageMeter()

    result = [[],[],[],[],[]] # for plotting

    if training:
        model.train()
    else:
        model.eval()
    end = time.time()
    for step, (data, ref, est) in enumerate(dataloader):
        batch_size = args.batch_size
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if args.use_cuda:
            data = data.cuda()
            ref = ref.cuda()
            est = est.cuda()
        TransformMatrix = forward(data)
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        _loss_total = 0
        _hinge_loss_total = 0
        _volume_loss_total = 0

        LHS = est-ref
        RHS = torch.abs(TransformMatrix)

        _hinge_loss = hinge_loss_function(LHS, RHS, alpha)
        _volume_loss = torch.sum(torch.abs(TransformMatrix),1)

        _hinge_loss = _hinge_loss.mean()
        _volume_loss = _volume_loss.mean()

        _loss = _hinge_loss + _lambda * _volume_loss
        _loss_total += _loss
        _hinge_loss_total += _hinge_loss
        _volume_loss_total += _volume_loss


        loss.update(_loss_total.item(), batch_size)
        prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss_total.item(), batch_size)
        volume_loss.update(_volume_loss_total.item(), batch_size)
        print(step, round(_loss_total.item(),2), round(_hinge_loss_total.item(),2), round(_volume_loss_total.item(),2))
        # l2_loss.update(_l2_loss.item(), batch_size)

        # if writer is not None and training:
        #     writer.add_scalar('loss', loss.val, global_step)
        #     writer.add_scalar('prec', prec.val, global_step)
        #     writer.add_scalar('Volume_loss', volume_loss.val, global_step)
        #     writer.add_scalar('Hinge_loss', hinge_loss.val, global_step)
        #     writer.add_scalar('L2_loss', l2_loss.val, global_step)

        time_str += 'other time: %.3f s\t'%(time.time()-end)
        c = time.time()
        if training:
            global_step += 1
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        time_str += 'backward time: %.3f s'%(time.time()-c)
        end = time.time()

    # print('Loss: %.3f, PREC: %.3f, HINGE_LOSS: %.3f, VOLUME_LOSS: %.3f, L2_loss: %.3f'%(loss.avg, prec.avg, hinge_loss.avg, volume_loss.avg, l2_loss.avg))

    # if writer is not None and not training:
    #     writer.add_scalar('loss', loss.avg, global_step)
    #     writer.add_scalar('prec', prec.avg, global_step)
    #     writer.add_scalar('Volume_loss', volume_loss.avg, global_step)
    #     writer.add_scalar('Hinge_loss', hinge_loss.avg, global_step)
    #     writer.add_scalar('L2_loss', l2_loss.avg, global_step)

    return result, loss.avg, prec.avg

def train_model(args):  
    from data import get_dataloader_autoland_dim
    train_loader, val_loader = get_dataloader_autoland_dim(args)
    from model import get_model_rect, get_model_rect2
    if args.dimension == 'x':
        model, forward = get_model_rect2(1, 2, 256, 256, 256)
    elif args.dimension == 'y':
        model, forward = get_model_rect(2, 2, 64, 64)
    elif args.dimension == 'z':
        model, forward = get_model_rect(2, 2, 64, 64)
    elif args.dimension == 'roll':
        model, forward = get_model_rect(2, 2, 64, 64)
    elif args.dimension == 'pitch':
        model, forward = get_model_rect(2, 2, 64, 64)
    elif args.dimension == 'yaw':
        model, forward = get_model_rect(2, 2, 64, 64)
    else:
        raise ValueError

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    os.system('mkdir '+args.log)
    os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), args.log))

    if args.use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # train_writer = SummaryWriter(args.log+'/train')
    # val_writer = SummaryWriter(args.log+'/val')

    best_loss = np.inf
    best_prec = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        print('Epoch %d'%(epoch))
        _, _, _ = trainval(model, forward, optimizer, epoch, train_loader, writer=None, training=True, alpha=args.alpha, _lambda=args._lambda)
        result_train, _, _ = trainval(model, forward, optimizer, epoch, train_loader, writer=None, training=False, alpha=args.alpha, _lambda=args._lambda)
        result_val, loss, prec = trainval(model, forward, optimizer, epoch, val_loader, writer=None, training=False, alpha=args.alpha, _lambda=args._lambda)
        epoch += 1
        # if prec > best_prec:
        if loss < best_loss:
            best_loss = loss
            print(best_loss)
            # best_prec = prec
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, filename=f"checkpoint_{args.dimension}_{start_time_str}_{epoch}.pth.tar")

if __name__ == "__main__":
    script_dir = os.path.realpath(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--system', type=str, default='autoland', help='Name of the dynamical system.')
    parser.add_argument('--lambda', dest='_lambda', type=float, default=1.0, help='lambda for balancing the two loss terms.')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='Hyper-parameter in the hinge loss.')
    parser.add_argument('--N_x0', type=int, default=10, help='Number of samples for the initial state x0.')
    parser.add_argument('--layer1', type=int, default=64, help='Number of neurons in the first layer of the NN.')
    parser.add_argument('--layer2', type=int, default=64, help='Number of neurons in the second layer of the NN.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs for training.')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', default=os.path.join(script_dir, '../'), type=str, help='Path to the file for storing the generated training data set.')
    parser.add_argument('--label_dir', default=os.path.join(script_dir, '../'), type=str, help='Path to the file for storing the generated training data set.')

    parser.add_argument('--data_file_eval', default=os.path.join(script_dir, '../'), type=str, help='Path to the file for storing the generated evaluation data set.')

    parser.add_argument('--log', default=os.path.join(script_dir, '../NeuReach/log'), type=str, help='Path to the directory for storing the logging files.')
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Use this option to disable cuda, if you want to train the NN on CPU.')
    parser.set_defaults(use_cuda=True)

    parser.add_argument('--bs', dest='batch_size', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dimension', '-d', type=str, default='x')

    args = parser.parse_args()

    train_model(args)