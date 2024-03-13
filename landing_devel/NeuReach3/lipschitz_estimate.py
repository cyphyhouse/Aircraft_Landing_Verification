import numpy as np 

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
    
    return np.array([xr, yr, zr, psir, thetar, vr])

def u(state):
    # state: bs*N*6
    # return: bs*N*6 
    res = np.zeros(state.shape)
    for i in range(state.shape[0]):
        x,y,z,psi,theta,v = state[i] 
        xr, yr, zr, psir, thetar, vr = compute_ref(state[i])
        
        xe = np.cos(psi)*(xr-x)+np.sin(psi)*(yr-y)
        ye = -np.sin(psi)*(xr-x)+np.cos(psi)*(yr-y)
        ze = zr-z 
        psie = psir-psi 
        thetae = thetar-theta 
        ve = vr-v 

        a = 0.05*(np.sqrt((vr*np.cos(thetar)*np.cos(psir)+0.01*xe)**2+(vr*np.sin(thetar)+0.01*ze)**2)-v)
        beta = psie+vr*(0.01*ye+0.01*np.sin(psie))
        omega = thetae+0.001*ze
        res[i,:] = np.array([a,beta,omega,0,0,0])
    return res

def f(state,u):
    # state: bs*N*6
    # u: bs*N*6
    res = np.zeros(state.shape)
    for i in range(state.shape[0]):
        x,y,z,psi,theta,v = state[i,:]
        a, beta, omega, un1, un2, un3 = u[i,:]
        dx = v*np.cos(psi)*np.cos(theta)
        dy = v*np.sin(psi)*np.cos(theta) 
        dz = np.sin(theta)
        dpsi = beta 
        dtheta = omega 
        dv = a
        res[i,:] = np.array([dx, dy, dz, dpsi, dtheta, dv])
    return res

if __name__ == "__main__":

    N = 1000

    x = np.random.uniform(1500,3500, (N, 1))
    y = np.random.uniform(-100,100, (N, 1))
    z = np.random.uniform(140,0, (N, 1))
    yaw = np.random.uniform(-np.pi/6,np.pi/6, (N, 1))
    pitch = np.random.uniform(-0.17453292519 ,0.17453292519 , (N, 1))
    v = np.random.uniform(9,11, (N, 1))

    state = np.hstack((x,y,z,yaw,pitch,v))

    u_orig = u(state)
    u_perturbed = u_orig + np.random.uniform(-0.001,0.001,u_orig.shape)

    fxu_orig = f(state, u_orig)
    fxu_perturbed = f(state, u_perturbed)

    lip_all = np.linalg.norm(fxu_orig-fxu_perturbed, axis=1)/np.linalg.norm(u_orig-u_perturbed, axis=1)

    print(lip_all)
    print(lip_all.max())