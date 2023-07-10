import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from model import get_model_rect2, get_model_rect, get_model_rect3
from sklearn import preprocessing
import sys 
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
import polytope as pc
import pypoman as ppm

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

def plot_polytope_3d(A, b, ax = None, edgecolor = 'k', color = 'red', trans = 0.2):
    verts = np.array(ppm.compute_polytope_vertices(A, b))
    # compute the triangles that make up the convex hull of the data points
    hull = ConvexHull(verts)
    triangles = [verts[s] for s in hull.simplices]
    # combine co-planar triangles into a single face
    faces = Faces(triangles, sig_dig=1).simplify()
    # plot
    if ax == None:
        ax = a3.Axes3D(plt.figure())

    pc = a3.art3d.Poly3DCollection(faces,
                                    facecolor=color,
                                    edgecolor=edgecolor, alpha=trans)
    ax.add_collection3d(pc)
    # define view
    yllim, ytlim = ax.get_ylim()
    xllim, xtlim = ax.get_xlim()
    zllim, ztlim = ax.get_zlim()
    x = verts[:,0]
    x = np.append(x, [xllim+1, xtlim-1])
    y = verts[:,1]
    y = np.append(y, [yllim+1, ytlim-1])
    z = verts[:,2]
    z = np.append(z, [zllim+1, ztlim-1])
    # print(np.min(x)-1, np.max(x)+1, np.min(y)-1, np.max(y)+1, np.min(z)-1, np.max(z)+1)
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_zlim(np.min(z)-1, np.max(z)+1)

script_dir = os.path.realpath(os.path.dirname(__file__))

model_r_name_x = 'checkpoint_x_r_06-26_10-46-59_39.pth.tar'
model_c_name_x = 'checkpoint_x_c_06-26_10-46-59_39.pth.tar'
model_r_name_y = 'checkpoint_y_r_06-26_10-48-42_39.pth.tar'
model_c_name_y = 'checkpoint_y_c_06-26_10-48-42_39.pth.tar'
model_r_name_z = 'checkpoint_z_r_06-26_10-59-29_70.pth.tar'
model_c_name_z = 'checkpoint_z_c_06-26_10-59-29_70.pth.tar'

model_r_x, forward_r_x = get_model_rect(1,1,64,64)
model_c_x, forward_c_x = get_model_rect(1,1,64,64)
model_r_y, forward_r_y = get_model_rect(2,1,64,64)
model_c_y, forward_c_y = get_model_rect(2,1,64,64)
model_r_z, forward_r_z = get_model_rect(2,1,64,64)
model_c_z, forward_c_z = get_model_rect(2,1,64,64)

# model, forward = get_model_rect(6,6,32,32)

model_r_x.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_r_name_x}'), map_location=torch.device('cpu'))['state_dict'])
model_c_x.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_c_name_x}'), map_location=torch.device('cpu'))['state_dict'])
model_r_y.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_r_name_y}'), map_location=torch.device('cpu'))['state_dict'])
model_c_y.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_c_name_y}'), map_location=torch.device('cpu'))['state_dict'])
model_r_z.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_r_name_z}'), map_location=torch.device('cpu'))['state_dict'])
model_c_z.load_state_dict(torch.load(os.path.join(script_dir, f'./log/{model_c_name_z}'), map_location=torch.device('cpu'))['state_dict'])

# data = torch.FloatTensor([-2936.190526247269, 23.028459769554445, 56.49611197902172, 0.041778978197086855, 0.0498730895584773, -0.013122412801362213])

data_path = os.path.join(script_dir, './ground_truth.npy')
label_path = os.path.join(script_dir, './estimation.npy')

data_orig = np.load(data_path)
label = np.load(label_path)

data_x = data_orig[:,0:1]
label_x = label[:,0:1]
ref_x = data_x.squeeze()
data_y = data_orig[:,[0,1]]
label_y = label[:, 1:2]
ref_y = data_orig[:,1:2].squeeze()
data_z = data_orig[:,[0,2]]
label_z = label[:, 2:3]
ref_z = data_orig[:,2:3].squeeze()

data_x_tensor = torch.FloatTensor(data_x)
label_x_tensor = torch.FloatTensor(label_x)
data_y_tensor = torch.FloatTensor(data_y)
label_y_tensor = torch.FloatTensor(label_y)
data_z_tensor = torch.FloatTensor(data_z)
label_z_tensor = torch.FloatTensor(label_z)

res_r_x = model_r_x.forward(data_x_tensor).detach().numpy()
res_r_x = np.abs(res_r_x)
res_c_x = model_c_x.forward(data_x_tensor).detach().numpy()
res_r_y = model_r_y.forward(data_y_tensor).detach().numpy()
res_r_y = np.abs(res_r_y)
res_c_y = model_c_y.forward(data_y_tensor).detach().numpy()
res_r_z = model_r_z.forward(data_z_tensor).detach().numpy()
res_r_z = np.abs(res_r_z)
res_c_z = model_c_z.forward(data_z_tensor).detach().numpy()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(ref_x, ref_y, ref_z, linewidth=5, color='g')
ax.plot(label_x.squeeze(), label_y.squeeze(), label_z.squeeze(), linewidth=5, color='b')

for i in range(res_r_x.shape[0]):
    print(i)

    x_lb = res_c_x[i,0] - res_r_x[i,0] 
    x_ub = res_c_x[i,0] + res_r_x[i,0]
    y_lb = res_c_y[i,0] - res_r_y[i,0] 
    y_ub = res_c_y[i,0] + res_r_y[i,0]
    z_lb = res_c_z[i,0] - res_r_z[i,0] 
    z_ub = res_c_z[i,0] + res_r_z[i,0]
    box = [
        [x_lb, x_ub],
        [y_lb, y_ub],
        [z_lb, z_ub],
    ]
    poly = pc.box2poly(np.array(box))
    plot_polytope_3d(poly.A, poly.b, ax, trans=0.01, edgecolor='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
# res = scaler_label_train.inverse_transform(res)

# if dim == 'x':
#     plt.figure()
#     plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     plt.plot(data[:,0], res_c+res_r,'r*')
#     plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     plt.legend()
#     plt.xlabel("ground truth x")
#     plt.ylabel('estimated x')
#     # plt.savefig('surrogate_bound_x.png')
# elif dim == 'y':
#     # plt.figure()
#     # plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     # plt.plot(data[:,0], res_c+res_r,'r*')
#     # plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     # plt.legend()
#     # plt.xlabel("ground truth x")
#     # plt.ylabel('estimated y')
#     # plt.savefig('surrogate_bound_y.png')
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:,0], data[:,1], label[:,0], c='b', marker='*')
#     ax.scatter(data[:,0], data[:,1], res_c+res_r, c='r', marker='*')
#     ax.scatter(data[:,0], data[:,1], res_c-res_r, c='r', marker='*')
#     ax.set_xlabel('ground truth x')
#     ax.set_ylabel('ground truth y')
#     ax.set_zlabel('estimate y')
# elif dim == 'z':
#     plt.figure()
#     plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     plt.plot(data[:,0], res_c+res_r,'r*')
#     plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     plt.legend()
#     plt.xlabel("ground truth x")
#     plt.ylabel('estimated z')
#     # plt.savefig('surrogate_bound_z.png')
# elif dim == 'roll':
#     plt.figure()
#     plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     plt.plot(data[:,0], res_c+res_r,'r*')
#     plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     plt.legend()
#     plt.xlabel("ground truth roll")
#     plt.ylabel('estimated roll')
#     # plt.savefig('surrogate_bound_roll.png')
# elif dim == 'pitch':
#     plt.figure()
#     plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     plt.plot(data[:,0], res_c+res_r,'r*')
#     plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     plt.legend()
#     plt.xlabel("ground truth pitch")
#     plt.ylabel('estimated pitch')
#     # plt.savefig('surrogate_bound_pitch.png')
# elif dim == 'yaw':
#     plt.figure()
#     plt.plot(data[:,0], label[:,0],'b*', label='estimated state')
#     plt.plot(data[:,0], res_c+res_r,'r*')
#     plt.plot(data[:,0], res_c-res_r,'r*', label='surrogate bound')
#     plt.legend()
#     plt.xlabel("ground truth yaw")
#     plt.ylabel('estimated yaw')
#     # plt.savefig('surrogate_bound_yaw.png')

# plt.show()
# # label = torch.FloatTensor([-2929.1353320511444, 20.64578387453148, 58.76066196314996, 0.04082988026075878, 0.05136111452277414, -0.012049659860212891])
# # print(data)
# # print(res)
# # print(label)
    