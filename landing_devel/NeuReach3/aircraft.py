from numpy import pi 
import numpy as np 
import torch 
from contraction_metric import Metric
from torch.autograd import grad

def compute_ref(state, approaching_angle = 3):
    x,y,z,psi,theta,v = state
    xinit = -3000
    yinit = 0
    zinit = 120 
    psiinit = 0
    thetainit = -np.deg2rad(3)
    vinit = 10 
    k = np.tan(approaching_angle*np.pi/180)
    # xr = xinit + vinit*t 
    # yr = yinit 
    # zr = zinit+k*vinit*t 
    # psir = psiinit 
    # thetar = thetainit 
    # vr = vinit 
    xr = x 
    yr = yinit 
    zr = -k*(x+3000)+120
    psir = psiinit 
    thetar = thetainit 
    vr = vinit 
    
    return torch.FloatTensor([xr, yr, zr, psir, thetar, vr])

def f(state,u):
    # state: bs*N*6
    # u: bs*N*6
    res = torch.zeros(state.shape)
    for i in range(state.shape[0]):
        x,y,z,psi,theta,v = state[i,:]
        a, beta, omega, un1, un2, un3 = u[i,:]
        dx = v*torch.cos(psi)*torch.cos(theta)
        dy = v*torch.sin(psi)*torch.cos(theta) 
        dz = torch.sin(theta)
        dpsi = beta 
        dtheta = omega 
        dv = a
        res[i,:] = torch.FloatTensor([dx, dy, dz, dpsi, dtheta, dv])
    return res

def u(state):
    # state: bs*N*6
    # return: bs*N*6 
    res = torch.zeros(state.shape)
    for i in range(state.shape[0]):
        x,y,z,psi,theta,v = state[i] 
        xr, yr, zr, psir, thetar, vr = compute_ref(state[i])
        
        xe = torch.cos(psi)*(xr-x)+torch.sin(psi)*(yr-y)
        ye = -torch.sin(psi)*(xr-x)+torch.cos(psi)*(yr-y)
        ze = zr-z 
        psie = psir-psi 
        thetae = thetar-theta 
        ve = vr-v 

        a = 0.05*(torch.sqrt((vr*torch.cos(thetar)*torch.cos(psir)+0.01*xe)**2+(vr*torch.sin(thetar)+0.01*ze)**2)-v)
        beta = psie+vr*(0.01*ye+0.01*torch.sin(psie))
        omega = thetae+0.001*ze
        res[i,:] = torch.FloatTensor([a,beta,omega,0,0,0])
    return res

def pfpx(state: torch.FloatTensor):
    # state: N*6
    # return: N*6*6
    res = torch.zeros((state.shape[0], 6, 6))
    for i in range(res.shape[0]):
        x,y,z,psi,theta,v = state[i,:] 
        pfpx = torch.FloatTensor([0,0,0,0,0,0]).reshape((-1,1))
        pfpy = torch.FloatTensor([0,0,0,0,0,0]).reshape((-1,1))
        pfpz = torch.FloatTensor([0,0,0,0,0,0]).reshape((-1,1))
        pfppsi = torch.FloatTensor([
            -v*torch.sin(psi)*torch.cos(theta), 
            v*torch.cos(psi)*torch.cos(theta), 
            0,0,0,0
        ]).reshape((-1,1))
        pfptheta = torch.FloatTensor([
            -v*torch.cos(psi)*torch.sin(theta),
            -v*torch.sin(psi)*torch.sin(theta), 
            torch.cos(theta),
            0, 0, 0
        ]).reshape((-1,1))
        pfpv = torch.FloatTensor([
            torch.cos(psi)*torch.cos(theta),
            torch.sin(psi)*torch.cos(theta),
            0,0,0,0
        ]).reshape((-1,1))
        tmp = torch.hstack(
            (pfpx, pfpy, pfpz, pfppsi, pfptheta, pfpv)
        )
        res[i,:,:] = tmp
    return res.cuda()

def pfpu(state):
    # state: N*6
    # return: N*6*6
    template = torch.FloatTensor([
        [0,0,0,0,0,0],
        [0,0,0,0,0,0], 
        [0,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [1,0,0,0,0,0]
    ]).reshape(1,6,6).cuda()
    res = template.repeat(state.shape[0], 1, 1)
    res = res+0*state.sum()
    return res

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def dmdt(M, state):
    dxdt = f(state, u(state)).cuda()
    dmdx = Jacobian_Matrix(M,state).cuda()

    res = (dmdx*dxdt.view(state.shape[0],1,1,-1)).sum(dim=3)
    return res

def Lpd_sample(LHS: torch.FloatTensor, num_sample = 10000):
    z = torch.randn(num_sample, LHS.size(-1)).cuda()
    z = z/z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(LHS) * z.view(1,num_sample,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def forward(metric: Metric, data: torch.FloatTensor, lamb):
    M = metric(data).cuda()

    pfx = pfpx(data).cuda()
    pfu = pfpu(data).cuda()

    dmx = dmdt(M, data).cuda()

    LHS = -(pfx.transpose(1,2).matmul(M) + M.matmul(pfx) + \
        pfu.transpose(1,2).matmul(M) + M.matmul(pfu) + \
        dmx +lamb*M)
    
    loss = Lpd_sample(LHS)
    return loss 


if __name__ == "__main__":
    device = torch.device('cuda')
    metric = Metric(6, 64)
    metric = metric.to(device)

    optimizer = torch.optim.Adam(metric.parameters(), lr=0.001)
    lamb = 1 
    N = 1000
    _iter = 10000

    x = np.random.uniform(1500,3500, (N, 1))
    y = np.random.uniform(-100,100, (N, 1))
    z = np.random.uniform(140,0, (N, 1))
    yaw = np.random.uniform(-np.pi/6,np.pi/6, (N, 1))
    pitch = np.random.uniform(-0.17453292519 ,0.17453292519 , (N, 1))
    v = np.random.uniform(9,11, (N, 1))

    data = np.hstack((x,y,z,yaw,pitch,v))

    data = torch.FloatTensor(data).cuda()
    data = data.requires_grad_()

    for i in range(_iter):
        loss = forward(metric, data, lamb)
        print(i, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # pfpx = 
    # pfpu = 

# def aircraft_dynamics(self, state, t):
#         # This function are the "tracking" dynamics used for the dubin's aircraft
#         x,y,z,heading, pitch, velocity = self.estimated_state
#         headingInput, pitchInput, accelInput = self.cst_input

#         heading = heading%(2*pi)
#         if heading > pi:
#             heading = heading - 2*pi
#         pitch = pitch%(2*pi)
#         if pitch > pi:
#             pitch = pitch - 2*pi


#         xref, yref, zref, headingref, pitchref, velref = self.goal_state
#         # print(f"Goal state: {xref}; Estimate state: {x}")
#         x_err = np.cos(heading)*(xref - x) + np.sin(heading)*(yref - y)
#         y_err = -np.sin(heading)*(xref - x) + np.cos(heading)*(yref - y)
#         z_err = zref - z
#         heading_err = headingref - heading

#         new_vel_xy = velref*np.cos(pitchref)*np.cos(heading_err)+self.K1[0]*x_err
#         new_heading_input = heading_err + velref*(self.K1[1]*y_err + self.K1[2]*np.sin(heading_err))
#         new_vel_z = velref*np.sin(pitchref)+self.K1[3]*z_err
#         new_vel = np.sqrt(new_vel_xy**2 + new_vel_z**2)

#         headingInput = new_heading_input
#         accelInput = self.K2[0]*(new_vel - velocity)
#         pitchInput = (pitchref - pitch) + (self.K2[1]*z_err)

#         # if 'SAFETY' in str(mode[0]):
#         #     if velocity <= 70:
#         #         accelInput = 0
#         #     else:
#         #         accelInput = -10

#         # Time derivative of the states
#         # dxdt = velocity*cos(heading)*cos(pitch)
#         # dydt = velocity*np.sin(heading)*cos(pitch)
#         # dzdt = velocity*np.sin(pitch)
#         dxdt = self.initial_state[5]*np.cos(self.initial_state[3])*np.cos(self.initial_state[4])
#         dydt = self.initial_state[5]*np.sin(self.initial_state[3])*np.cos(self.initial_state[4])
#         dzdt = self.initial_state[5]*np.sin(self.initial_state[4])
#         dheadingdt = headingInput
#         dpitchdt = pitchInput
#         dveldt = accelInput

#         print(dxdt, dydt, dzdt)

#         accel_max = 10
#         heading_rate_max = pi/18
#         pitch_rate_max = pi/18

#         if abs(dveldt)>accel_max:
#             dveldt = np.sign(dveldt)*accel_max
#         if abs(dpitchdt)>pitch_rate_max*1:
#             dpitchdt = np.sign(dpitchdt)*pitch_rate_max
#         if abs(dheadingdt)>heading_rate_max:
#             dheadingdt = np.sign(dheadingdt)*heading_rate_max

#         return [dxdt, dydt, dzdt, dheadingdt, dpitchdt, dveldt]