import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt 

A = np.eye(7)
noise = [
    [-0.08203863, -0.03201971, -0.02689552, -0.09407139,  0.07471189, -0.056743  ,  0.00332492],
    [ 0.04063152, -0.06869586,  0.05745584,  0.03321152, -0.04550705, -0.05093219, -0.01757601],
    [ 0.00814388,  0.00862197, -0.06264512, -0.06197693, -0.07459895, -0.02027042, -0.01162196],
    [ 0.05222916,  0.03141921,  0.0919425 ,  0.05145495,  0.09485368,  0.09970963, -0.05110597],
    [ 0.09672396, -0.06652737, -0.07418409,  0.01972249,  0.05946392,  0.01262568, -0.08198125],
    [ 0.04961994,  0.01845722,  0.01250611, -0.05911221, -0.06518864,  0.01816719, -0.04239076],
    [ 0.07823648, -0.08568998,  0.07475148, -0.05854442,  0.08695816,  0.02317259, -0.00653621],
]
A = A+noise
# A = A + np.random.uniform(-0.1,0.1,(7,7))
d = 0
# d = np.random.uniform(-0.1, 0.1, 7)
print(A)
print(d)
def f(x,t):
    x1, x2, x3,\
    x4, x5, x6,\
    x7 = x 

    dx1 = 1.4*x3-0.9*x1 
    dx2 = 2.5*x5-1.5*x2 
    dx3 = 0.6*x7-0.8*x2*x3 
    dx4 = 2-1.3*x3*x4 
    dx5 = 0.7*x1-x4*x5 
    dx6 = 0.3*x1-3.1*x6 
    dx7 = 1.8*x6-1.5*x2*x7  

    return [dx1,dx2,dx3,dx4,dx5,dx6,dx7]

def fA(x,t):
    x = x + np.sin(x)
    x1, x2, x3,\
    x4, x5, x6,\
    x7 = x 

    dx1 = 1.4*x3-0.9*x1 
    dx2 = 2.5*x5-1.5*x2 
    dx3 = 0.6*x7-0.8*x2*x3 
    dx4 = 2-1.3*x3*x4 
    dx5 = 0.7*x1-x4*x5 
    dx6 = 0.3*x1-3.1*x6 
    dx7 = 1.8*x6-1.5*x2*x7  

    return [dx1,dx2,dx3,dx4,dx5,dx6,dx7]

if __name__ == "__main__":
    x1 = [1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]
    x2 = [1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]
    Tmax = 100.0
    dt = 0.005 
    T = np.round(np.arange(0.0, Tmax+dt/2, dt),8)

    res1 = odeint(f, x1, T)

    res2 = odeint(fA, x2, T)

    plt.plot(T, res1[:,4], 'r')
    plt.plot(T, res2[:,4], 'b')

    plt.show()