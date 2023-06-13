import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
import copy
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt 

from verse.analysis.NeuReach.utils import AverageMeter
from verse.analysis.NeuReach.data import get_dataloader
from verse.analysis.NeuReach.model import get_model_rect

import sys
sys.path.append('systems')

import argparse

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

default_params = {
    "seed": 0,
    "N_X0": 5,
    "N_x0": 100,
    "N_t": 100,
    "data_file_train": 'train.pklz',
    "batch_size": 256,
    "num_test": 10,
    "data_file_eval": 'eval.pklz',
    "use_cuda": False,
    "layer1": 64,
    "layer2": 64,
    "epochs": 10,
    "learning_rate": 0.05,
    "lr_step": 10,
    "alpha": 0.0001,
    "_lambda": 0.03,
    "r": 0.1,
}

def adjust_learning_rate(optimizer, epoch, learning_rate, lr_step):
    """Sets the learning rate to the initial LR decayed by 10 every * epochs"""
    lr = learning_rate * (0.1 ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, log, filename='checkpoint.pth.tar'):
    filename = log + '/' + filename
    torch.save(state, filename)

def hinge_loss_function(LHS, RHS, alpha):
    res = LHS - RHS + alpha
    res[res<0] = 0
    res = res.sum(dim=1)
    return res

global_step = 0

def trainval(
    epoch, dataloader, writer, training, model, optimizer, forward, 
    use_cuda, alpha, _lambda
):
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
    for step, (X0, t, ref, xt) in enumerate(dataloader):
        batch_size = X0.size(0)
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if use_cuda:
            X0 = X0.cuda()
            t = t.cuda()
            ref = ref.cuda()
            xt = xt.cuda()
        TransformMatrix = forward(torch.cat([X0,t], dim=1))
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        DXi = xt - ref
        # LHS = ((torch.matmul(TransformMatrix, DXi.view(batch_size,-1,1)).view(batch_size,-1)) ** 2).sum(dim=1)
        # RHS = torch.ones(LHS.size()).type(DXi.type())
        LHS = torch.abs(DXi)
        RHS = torch.abs(TransformMatrix)
        # _hinge_loss = hinge_loss_function(LHS, RHS, args)
        # _volume_loss = -torch.log((TransformMatrix + 0.01 * torch.eye(TransformMatrix.shape[-1]).unsqueeze(0).type(X0.type())).det().abs())
        _hinge_loss = hinge_loss_function(LHS, RHS, alpha)
        _volume_loss = torch.prod(torch.abs(TransformMatrix),1)


        # mask = _hinge_loss > 0
        # _volume_loss[mask] = 0.0
        _hinge_loss = _hinge_loss.mean()
        _volume_loss = _volume_loss.mean()

        # CY2 = torch.sqrt(LHS)
        # Y2 = torch.sqrt((DXi.view(batch_size,-1) ** 2).sum(dim=1))
        # _l2_loss = (torch.abs((CY2 - 1)) * Y2 / CY2).mean()

        _loss = _hinge_loss + _lambda * _volume_loss
        _loss *= 10

        loss.update(_loss.item(), batch_size)
        prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss.item(), batch_size)
        volume_loss.update(_volume_loss.item(), batch_size)
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

def NeuReach_rect(
    config,
    seed = 0,
    N_X0 = 5, N_x0 = 100, N_t = 100, data_file_train = 'train.pklz', batch_size = 256,
    num_test = 10, data_file_eval = 'eval.pklz',
    use_cuda = False,
    layer1 = 64, layer2 = 64,
    epochs = 10,
    learning_rate = 0.05, lr_step = 10,
    alpha = 0.0001, _lambda = 0.03,
):
    np.random.seed(seed)
    torch.manual_seed(seed)


    # os.system('mkdir '+log)
    # os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), log))
    # os.system('cp *.py '+log)
    # os.system('cp -r systems/ '+log)
    # os.system('cp -r ODEs/ '+log)

    # config = importlib.import_module('system_'+args.system)
    model, forward = get_model_rect(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1, config, layer1, layer2)
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_dataloader(
        config, N_X0, N_x0, N_t, data_file_train, batch_size,
        num_test, data_file_eval
    )

    # train_writer = SummaryWriter(log+'/train')
    # val_writer = SummaryWriter(log+'/val')

    best_loss = np.inf
    best_prec = 0

    res = model.state_dict()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate, lr_step)
        # train for one epoch
        # print('Epoch %d'%(epoch))
        _, _, _ = trainval(epoch, train_loader, writer=None, training=True, model=model, optimizer=optimizer, forward=forward, alpha=alpha, _lambda=_lambda, use_cuda = use_cuda)
        result_train, _, _ = trainval(epoch, train_loader, writer=None, training=False, model=model, optimizer=optimizer, forward=forward, alpha=alpha, _lambda=_lambda, use_cuda = use_cuda)
        result_val, loss, prec = trainval(epoch, val_loader, writer=None, training=False, model=model, optimizer=optimizer, forward=forward, alpha=alpha, _lambda=_lambda, use_cuda = use_cuda)
        epoch += 1
        # if prec > best_prec:
        if loss < best_loss:
            best_loss = loss
            # best_prec = prec
            # save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, log)
            res = model.state_dict()
    return res


class SysConfig:
    def __init__(self, init, mode, time_bound, time_step, sim_func, r, track_map = None):
        self.init = init
        self.TMAX = time_bound 
        self.dt = time_step 
        self.sim_func = sim_func 
        self.mode = mode 
        self.r = r
        self.track_map = track_map

    def sample_X0(self):
        X0 = []
        for i in range(0,len(self.init[0])):
            X0.append(self.init[0][i]) 
            X0.append(self.init[1][i])
        # X0.append(self.r)
        return np.array(X0)

    def sample_t(self):
        return (np.random.randint(int(self.TMAX/self.dt))+1) * self.dt

    def sample_x0(self, X0):
        x0 = []
        for i in range(0,X0.shape[0]-1,2):
            low = X0[i]
            high = X0[i+1]
            val = np.random.uniform(low-self.r, high+self.r)
            x0.append(val)
        return np.array(x0)

    def simulate(self, x0):
        if isinstance(x0,np.ndarray):
            x0 = x0.tolist()
        return self.sim_func(self.mode, x0, self.TMAX, self.dt, self.track_map)

    def get_init_center(self, X0):
        center = []
        for i in range(0,X0.shape[0]-1,2):
            low = X0[i]
            high = X0[i+1]
            center.append((low+high)/2)
        return np.array(center)

    def get_X0_normalization_factor(self):
        mean = np.zeros(self.sample_X0().shape)
        std = np.ones(self.sample_X0().shape)
        return [mean, std]


def calculate_bloated_tube_NeuReach(
    mode, init, time_bound, time_step, sim_func, track_map = None, params = {}
):
    print(init)
    tmp = copy.deepcopy(default_params)
    tmp.update(params)
    seed = tmp["seed"]
    N_X0 = tmp["N_X0"]
    N_x0 = tmp["N_x0"]
    N_t = tmp["N_t"]
    data_file_train = tmp["data_file_train"]
    batch_size = tmp["batch_size"]
    num_test = tmp["num_test"]
    data_file_eval = tmp["data_file_eval"]
    use_cuda = tmp["use_cuda"]
    layer1 = tmp["layer1"]
    layer2 = tmp["layer2"]
    epochs = tmp["epochs"]
    learning_rate = tmp["learning_rate"]
    lr_step = tmp["lr_step"]
    alpha = tmp["alpha"]
    _lambda = tmp["_lambda"]
    r = tmp["r"]

    config = SysConfig(init, mode, time_bound, time_step, sim_func, r, track_map)
    best_state_dict = NeuReach_rect(
        config, 
        seed = seed,
        N_X0 = N_X0, N_x0 = N_x0, N_t = N_t, data_file_train = data_file_train, batch_size = batch_size,
        num_test = num_test, data_file_eval = data_file_eval,
        use_cuda = use_cuda,
        layer1 = layer1, layer2 = layer2,
        epochs = epochs,
        learning_rate = learning_rate, lr_step = lr_step,
        alpha = alpha, _lambda = _lambda
    )
    model_ours, forward_ours = get_model_rect(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1,  config, layer1, layer2)
    model_ours.load_state_dict(best_state_dict)

    X0 = config.sample_X0()
    center = config.get_init_center(X0)
    ref = config.simulate(center)
    num_t = ref.shape[0]
    res = []
    for i in range(num_t):
        tmp = torch.tensor(X0.tolist()+[ref[i,0]]).view(1,-1).float()
        P = forward_ours(tmp)
        P = P.squeeze(0)
        res.append(P.cpu().detach().numpy())
    
    reachtube = []
    for i in range(num_t - 1) :
        tmp_center = np.vstack((ref[i,1:],ref[i,1:], ref[i+1,1:],ref[i+1,1:]))
        tmp_radius = np.vstack((res[i],-res[i],res[i+1],-res[i+1]))
        lower_bound = np.insert(np.min(tmp_center+tmp_radius,axis=0),0,ref[i,0]).tolist()
        upper_bound = np.insert(np.max(tmp_center+tmp_radius,axis=0),0,ref[i+1,0]).tolist()
        reachtube.append(lower_bound)
        reachtube.append(upper_bound)
    return np.array(reachtube)

if __name__ == "__main__":
    partition = ((-0.7, 0.10471975511965977), (-0.5, 0.17453292519943295))
    # partition = ((-0.9, -0.17453292519943295), (-0.7, -0.10471975511965977))


    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--system', type=str,
                            default='jetengine', help='Name of the dynamical system.')
    parser.add_argument('--lambda', dest='_lambda', type=float, default=0.03, help='lambda for balancing the two loss terms.')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.0001, help='Hyper-parameter in the hinge loss.')
    parser.add_argument('--N_X0', type=int, default=5, help='Number of samples for the initial set X0.')
    parser.add_argument('--N_x0', type=int, default=100, help='Number of samples for the initial state x0.')
    parser.add_argument('--N_t', type=int, default=100, help='Number of samples for the time instant t.')
    parser.add_argument('--layer1', type=int, default=64, help='Number of neurons in the first layer of the NN.')
    parser.add_argument('--layer2', type=int, default=64, help='Number of neurons in the second layer of the NN.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--data_file_train', default='train.pklz', type=str, help='Path to the file for storing the generated training data set.')
    parser.add_argument('--data_file_eval', default='eval.pklz', type=str, help='Path to the file for storing the generated evaluation data set.')
    parser.add_argument('--log', type=str, default='', help='Path to the directory for storing the logging files.')
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Use this option to disable cuda, if you want to train the NN on CPU.')
    parser.set_defaults(use_cuda=True)

    parser.add_argument('--bs', dest='batch_size', type=int, default=256)
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--x_lower', type=float)
    parser.add_argument('--x_upper', type=float)
    parser.add_argument('--y_lower', type=float)
    parser.add_argument('--y_upper', type=float)
    parser.add_argument('--theta_lower', type=float)
    parser.add_argument('--theta_upper', type=float)
    parser.add_argument('--TMAX', type=float)
    parser.add_argument('--dt', type=float)

    args = parser.parse_args()
    args.log = "log_lanetracking"
    args.data_file_train = "lanetracking_train.pklz"
    args.data_file_eval = "lanetracking_eval.pklz"
    args.x_lower = 0
    args.x_upper = 0
    # args.y_lower = -0.7
    # args.y_upper = -0.5
    args.y_lower = partition[0][0]
    args.y_upper = partition[1][0]
    # args.theta_lower = 0.10471975511965977
    # args.theta_upper = 0.17453292519943295
    args.theta_lower = partition[0][1]
    args.theta_upper = partition[1][1]
    args.r = 0.19897
    args.TMAX = 0.5
    args.dt = 0.1
    args.use_cuda = False

    NeuReach_rect(args)

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes', axisbelow=True)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.15,
        left=0.11,
        right=1.0,
        hspace=0.2,
        wspace=0.2)

    import sys
    sys.path.append('systems')
    sys.path.append('.')

    from model import get_model_rect
    from model_dryvr import get_model as get_model_dryvr

    import argparse

    np.random.seed(1024)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--system', type=str,
                            default='lanetracking')
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--layer1', type=int, default=64)
    parser.add_argument('--layer2', type=int, default=64)
    parser.add_argument('--pretrained_ours', type=str)
    parser.add_argument('--pretrained_dryvr', type=str)

    args = parser.parse_args()
    args.use_cuda = False 
    args.pretrained_ours = 'log_lanetracking/checkpoint.pth.tar'
    args.system = 'lane_tracking_new'

    config = LaneTrackingSystem(
        partition[0][0],
        partition[1][0],
        partition[0][1],
        partition[1][1],
        0.19
    )

    model_ours, forward_ours = get_model_rect(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1,  config, args)
    if args.use_cuda:
        model_ours = model_ours.cuda()
    else:
        model_ours = model_ours.cpu()
    model_ours.load_state_dict(torch.load(args.pretrained_ours)['state_dict'])

    # model_dryvr, forward_dryvr = get_model_dryvr(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1)
    # model_dryvr.load_state_dict(torch.load(args.pretrained_dryvr)['state_dict'])

    def ellipsoid_surface_2D(P):
        K = 100
        thetas = np.linspace(0, 2 * np.pi, K)
        points = []
        for i, theta in enumerate(thetas):
            point = np.array([np.cos(theta), np.sin(theta)])
            points.append(point)
        points = np.array(points)
        points = np.linalg.inv(P).dot(points.T)
        return points[0,:], points[1,:]

    def rectangular_surface_2D(P):
        x = [-P[0],P[0],P[0],-P[0],-P[0]]
        y = [-P[1],-P[1],P[1],P[1],-P[1]]
        return x,y

    benchmark_name = args.system

    # for l in range(6):
    #     for m in range(3):
    # l,m = 0,0
    # center = np.array([0,-0.75+l*0.3,(-6+m*6)*np.pi/180,-0.75+l*0.3,(-6+m*6)*np.pi/180])
    # r = ((np.eye(5)+np.array([
    #         [0,0,0,0,0],
    #         [0,0,0,0,0],
    #         [0,0,0,0,0],
    #         [0,0,0,3,0],
    #         [0,0,0,0,3]
    #     ]))*(0.05**2)).flatten()

    # print(center, r)

    X0 = config.sample_X0()
    center = config.get_init_center(X0)

    traces = []
    # ref trace
    ref = config.simulate(center)
    traces.append(np.array(ref))

    # calculate the reachset using the trained model
    reachsets_ours = []
    # reachsets_dryvr = []
    for idx_t in range(1, ref.shape[0]):
        tmp = torch.tensor(X0.tolist()+[ref[idx_t, 0]]).view(1,-1).float()
        if args.use_cuda:
            tmp = tmp.cuda()
        P = forward_ours(tmp)
        P = P.squeeze(0)
        reachsets_ours.append([ref[idx_t, 1:], P.cpu().detach().numpy()])
    # idx_t = ref.shape[0]-1
    # tmp = torch.tensor(X0.tolist()+[ref[idx_t, 0]]).view(1,-1).float()
    # print(tmp)
    # P = forward_ours(tmp)
    # P = P.squeeze(0)
    # reachsets_ours.append([ref[idx_t, 1:], P.cpu().detach().numpy()])

    print(len(reachsets_ours))
        # P = forward_dryvr(tmp)
        # P = P.squeeze(0)
        # reachsets_dryvr.append([ref[idx_t, 1:], P])

    # plot the ref trace
    plt.plot(ref[:,2], ref[:,3], 'r-')#, label='ref')

    # plot ellipsoids for each time step
    # for reachset_ours, reachset_dryvr in zip(reachsets_ours[::10], reachsets_dryvr[::10]):
    #     label = reachset_ours is reachsets_ours[0]
    #     c = reachset_ours[0]
    #     x,y = ellipsoid_surface_2D(reachset_ours[1])
    #     plt.plot(x+c[0], y+c[1], 'g-', markersize=1, label='NeuReach' if label else None)
    #     x,y = ellipsoid_surface_2D(reachset_dryvr[1])
    #     plt.plot(x+c[0], y+c[1], 'y-', markersize=1, label='DryVR' if label else None)

    for reachset_ours in [reachsets_ours[-1]]:
        label = reachset_ours is reachsets_ours[0]
        c = reachset_ours[0]
        print(reachset_ours[1])
        # projected_reachset = reachset_ours[1][1:3,1:3]
        projected_reachset = np.abs(reachset_ours[1][1:3])
        # x,y = ellipsoid_surface_2D(projected_reachset)
        x,y = rectangular_surface_2D(projected_reachset)
        # plt.plot(x+c[1], y+c[2], 'g-', markersize=1, label='NeuReach' if label else None)
        plt.plot(x+c[1], y+c[2], 'g-', markersize=1, label='NeuReach' if label else None)
        # x,y = ellipsoid_surface_2D(reachset_dryvr[1])
        # plt.plot(x+c[0], y+c[1], 'y-', marker

    sampled_traces = []

    # randomly sample some traces
    for i in range(10):
        # X0 = config.sample_X0()
        # X0 = np.concatenate((center, r))
        for _ in range(100):
            # n = len(center)
            # direction = np.random.randn(n)
            # direction = direction / np.linalg.norm(direction)

            # dist = np.random.rand()
            # x0 = center + direction * dist * r
            x0 = config.sample_x0(X0)
            _trace = config.simulate(x0)[:,1:]
            sampled_traces.append(_trace)

    _traces = np.array(sampled_traces)[:,1:,:]
    plt.plot(_traces[:,-1,1], _traces[:,-1,2], 'kx', markersize=1)

    # plt.xlim(-2, 1)
    # plt.ylim(-3, 0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(r'y')
    plt.ylabel(r'$\theta$')
    # plt.legend(loc='upper left')
    plt.title('Reachsets of lanetracking')
    plt.show()
    # plt.savefig('lanetracking.pdf')

