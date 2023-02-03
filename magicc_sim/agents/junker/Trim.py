import numpy as np
from numpy import linalg
import Param as P
from scipy.optimize import leastsq

# This code is based off the algorithms provided in the book "Small Unmanned Aircraft" by Randal W. Beard and Timothy W. McLain on 
# pages 278 - 284

# coefficient used to model the aerodynamics of the plane as a curved or flat object
def getSigma(alpha): 
    # The code is broken down into several steps to increase readability
    numerator = 1+np.exp(-P.M*(alpha-P.alpha_0))+np.exp(P.M*(alpha+P.alpha_0))
    den = ((1 + np.exp(-P.M*(alpha-P.alpha_0)))*(1+np.exp(P.M*(alpha+P.alpha_0))))
    sigma = numerator/den
    return sigma

# Returns the lift model coefficient
def getLiftModelCoeff(alpha,sigma): 
    C_L =  (1-sigma)*(P.C_L_0 + P.C_L_alpha*alpha) + sigma*(2.0*np.sign(alpha)*np.sin(alpha)**2.0*np.cos(alpha))
    return C_L

# Returns the drage model coefficient
def getDragModelCoeff(alpha): 
    AR = P.b**2.0/P.S
    C_D =P.C_D_p + (P.C_L_0 + P.C_L_alpha*alpha)**2.0/(np.pi*P.e*AR)
    return C_D

# Computes the lift coefficients 
def getLiftCoeff(alpha): 

    
    sigma = getSigma(alpha)              # Compute sigma    
    C_L = getLiftModelCoeff(alpha,sigma) # Compute Lift model Coeff     
    C_D = getDragModelCoeff(alpha)       # Compute drag model Coeff

    C_X =           -C_D*np.cos(alpha)  +          C_L*np.sin(alpha)
    C_X_q =         -P.C_D_q*np.cos(alpha) +            P.C_L_q*np.sin(alpha)
    C_X_delta_e =   -P.C_D_delta_e*np.cos(alpha) +      P.C_L_delta_e*np.sin(alpha)
    C_Z =           -C_D*np.sin(alpha) -           C_L*np.cos(alpha)
    C_Z_q =         -P.C_D_q*np.sin(alpha)-             P.C_L_q*np.cos(alpha)
    C_Z_delta_e =   -P.C_D_delta_e*np.sin(alpha) -      P.C_L_delta_e*np.cos(alpha)

    return C_X, C_X_q, C_X_delta_e, C_Z, C_Z_q, C_Z_delta_e

# Computes the body frame velocities
def getBodyFrameVelocities(alpha, beta,Va): 
    u = Va*np.cos(alpha)*np.cos(beta)
    v = Va*np.sin(beta)
    w = Va*np.sin(alpha)*np.cos(beta)
    return u, v, w

# Returns the pitch angel
def getPitchAngle(alpha,gamma):                  
    theta = alpha + gamma
    return theta

# Returns the angular rates 
def getAngularRates(phi,theta,Va,R):                 

    p = -Va/R*np.sin(theta)
    q = Va/R*np.sin(phi)*np.cos(theta)
    r = Va/R*np.cos(phi)*np.cos(theta)
    return p,q,r


# Returns the elevator coefficient
def getElevator(alpha,p,q,r,Va): 
    # The code is broken down into several steps to increase readability
    temp1 = (P.Jxz*(p**2.0 - r**2.0) + (P.Jx - P.Jz)*p*r)/(1.0/2.0*P.rho*Va**2.0*P.c*P.S)
    temp2 = temp1 - P.C_m_0 - P.C_m_alpha*alpha - P.C_m_q*(P.c*q/(2.0*Va))

    delta_e = temp2/P.C_m_delta_e
    return delta_e

# Returns the throttle coefficient
def getThrottle(u,v,w,p,q,r,Va,theta,C_X,C_X_q,C_X_delta_e,delta_e): 
    # The code is broken down into several steps to increase readability
    temp1 = 2.0*P.m*(-r*v + q*w + P.g*np.sin(theta))
    temp2 = -P.rho*(Va)**2.0*P.S*(C_X + C_X_q*P.c*q/(2.0*Va) + C_X_delta_e*delta_e)
    temp3 = (temp1 + temp2)/(P.rho*P.Sprop*P.Cprop*P.kmotor**2.0)
    temp4 = Va**2.0/P.kmotor**2.0
    delta_t = np.sqrt(temp3 + temp4)
    return delta_t

# Returns the aileron and rudder coefficients
def getAileronAndRudder(p,q,r,Va,beta): 
    # The code is broken down into several steps to increase readability
    matrix1 = np.matrix([[P.C_p_delta_a, P.C_p_delta_r], [P.C_r_delta_a, P.C_r_delta_r]])
    matrix1 = linalg.inv(matrix1)


    temp1 = (-P.Gamma1*p*q + P.Gamma2*q*r)/(0.5*P.rho*Va**2.0*P.S*P.b)
    temp2 = P.C_p_p*(P.b*p/(2.0*Va))
    temp3 = P.C_p_r*(P.b*r/(2.0*Va))
    temp4 = (-P.Gamma7*p*q + P.Gamma1*q*r)/(0.5*P.rho*Va**2.0*P.S*P.b)
    temp5 = P.C_r_p*(P.b*p/(2.0*Va))
    temp6 = P.C_r_r*(P.b*r/(2.0*Va))

    matrix2 = np.matrix([[temp1 - P.C_p_0 - P.C_p_beta*beta - temp2 - temp3],
        [temp4 - P.C_r_0 - P.C_r_beta*beta - temp5 - temp6]])

    controls = matrix1*matrix2

    delta_a = controls[0,0]
    delta_r = controls[1,0]

    return delta_a, delta_r




# Returns xtrim and utrim
def get_xtrim_utrim(alpha,beta,phi,Va,R,gamma):

    C_X, C_X_q, C_X_delta_e, C_Z, C_Z_q, C_Z_delta_e = getLiftCoeff(alpha)
    u,v,w = getBodyFrameVelocities(alpha, beta,Va)
    theta = getPitchAngle(alpha,gamma)
    p,q,r = getAngularRates(phi,theta,Va,R)
    delta_e = getElevator(alpha,p,q,r,Va)
    delta_t = getThrottle(u,v,w,p,q,r,Va,theta,C_X,C_X_q,C_X_delta_e,delta_e)
    delta_a, delta_r = getAileronAndRudder(p,q,r,Va,beta)

    return C_X, C_X_q, C_X_delta_e, C_Z, C_Z_q, C_Z_delta_e, u,v,w,theta,p,q,r,delta_e,delta_t,delta_a,delta_r



# Returns the state derivatives using x_trim and u_trim
def get_fx(alpha,beta,phi,Va,R,gamma):
        

    C_X, C_X_q, C_X_delta_e, C_Z, C_Z_q, C_Z_delta_e,u,v,w,theta,p,q,r,delta_e,delta_t,delta_a,delta_r = get_xtrim_utrim(alpha,beta,phi,Va,R,gamma)

    

    psi = 0.0


    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    cos_psi = np.cos(psi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    sin_psi = np.sin(psi)

    temp1 = P.rho*Va**2.0*P.S/(2.0*P.m)

    pndot = (cos_theta*cos_psi*u + (sin_phi*sin_theta*cos_psi - 
        cos_phi*sin_psi)*v + (cos_phi*sin_theta*cos_psi + 
        sin_phi*sin_psi)*w)

    pedot = (cos_theta*sin_psi*u + (sin_phi*sin_theta*sin_psi +
        cos_phi*cos_psi)*v + (cos_phi*sin_theta*sin_psi - 
        sin_phi*cos_psi)*w)

    hdot = (u*sin_theta - v*sin_phi*cos_theta -
        w*cos_phi*cos_theta)

    udot = (r*v - q*w - P.g*sin_theta + 
        temp1*(C_X + C_X_q*P.c*q/(2.0*Va) + C_X_delta_e*delta_e) +
        P.rho*P.Sprop*P.Cprop/(2.0*P.m)*((P.kmotor*delta_t)**2.0 - Va**2.0))

    vdot = (p*w - r*u + P.g*cos_theta*sin_phi + temp1*
        (P.C_Y_0 + P.C_Y_beta*beta + P.C_Y_p*P.b*p/(2.0*Va) + P.C_Y_r*P.b*r/(2.0*Va) +
         P.C_Y_delta_a*delta_a + P.C_Y_delta_r*delta_r))

    wdot = (q*u - p*v + P.g*cos_theta*cos_phi + temp1*(C_Z + 
        C_Z_q*P.c*q/(2.0*Va) + C_Z_delta_e*delta_e))

    phidot = p + q*sin_phi*np.tan(theta) + r*cos_phi*np.tan(theta)

    thetadot = q*cos_phi - r*sin_phi

    psidot = q*sin_phi/cos_theta + r*cos_phi/cos_theta
    

    pdot = (P.Gamma1*p*q - P.Gamma2*q*r + (1.0/2.0)*P.rho*Va**2.0*P.S*P.b*float(P.C_p_0 + P.C_p_beta*beta + P.C_p_p*P.b*p/(2.0*Va) + P.C_p_r*P.b*r/(2.0*Va) + P.C_p_delta_a*delta_a + P.C_p_delta_r*delta_r))

    qdot = (P.Gamma5*p*r - P.Gamma6*(p**2.0 - r**2.0) + P.rho*Va**2.0*P.S*P.c/(2.0*P.Jy)*
        (P.C_m_0 + P.C_m_alpha*alpha + P.C_m_q*P.c*q/(2.0*Va) + P.C_m_delta_e*delta_e))

    rdot = (P.Gamma7*p*q - P.Gamma1*q*r + (1.0/2.0)*P.rho*Va**2.0*P.S*P.b*
        (P.C_r_0 + P.C_r_beta*beta + P.C_r_p*P.b*p/(2.0*Va) + P.C_r_r*P.b*r/(2.0*Va) +
        P.C_r_delta_a*delta_a + P.C_r_delta_r*delta_r))

 
    fx = np.array([pndot,pedot,hdot,udot,vdot,wdot,phidot,thetadot,psidot,pdot,qdot,rdot])
    xtrim = np.array([0.0,                # pn
                      0.0,                # pe
                      0.0,                # pd
                      u,
                      v,
                      w,
                      phi,
                      theta,
                      psi,
                      p,
                      q,
                      r])

    utrim = np.array([delta_e,delta_t,delta_a,delta_r])


    return fx,xtrim,utrim

def printValues(fx, xtrim, utrim,plsq):

    alpha,beta,phi = plsq[0]

    print('\n\n x trim')
    print('pn',xtrim[0])
    print('pe',xtrim[1])
    print('pd',xtrim[2])
    print('u',xtrim[3])
    print('v',xtrim[4])
    print('w',xtrim[5])
    print('phi',xtrim[6])
    print('theta',xtrim[7])
    print('psi', xtrim[8])
    print('p',xtrim[9])
    print('q',xtrim[10])
    print('r',xtrim[11])

    print('\n\n u trim')
    print('delta e', utrim[0])
    print('delta a', utrim[2])
    print('delta r', utrim[3])
    print('delta t', utrim[1])

    print('\n\n xhatdot')
    print('pndot', fx[0])
    print('pedot', fx[1])
    print('hdot', fx[2])
    print('udot', fx[3])
    print('vdot', fx[4])
    print('wdot', fx[5])
    print('phidot', fx[6])
    print('thetadot', fx[7])
    print('psidot', fx[8])
    print('pdot', fx[9])
    print('qdot', fx[10])
    print('rdot', fx[11])

    print('beta ', beta)
    print('alpha ', alpha)

# Returns the error between desired xdot trim conditions and minimized xdot trim conditions
def residuals(p,y,x):
    alpha,beta,phi = p  # Current alpha,beta,phi
    Va, R, gamma = x    # Commanded action
    fx,xtrim,utrim = get_fx(alpha,beta,phi,Va,R,gamma) # Compute trim values based on the current 
                                                       # alpha, beta, and phi
    #err = y[2:] - fx[2:]
    err = y[1:] - fx[1:]  # Calcualtes the error between the desired xhot hat and the optimized xdot hat
    # print('err',err)
    return err

# Computes the minimized trim conditions iterativly 
def ComputeTrim(Va,R,gamma):
    # Desired xdot trim conditions
    xdotStar = np.array([0.0,                                 # pn dot
                         0.0,                                 # pe dot
                         Va*np.sin(gamma),        # h dot
                         0.0,                                 # u dot
                         0.0,                                 # v dot
                         0.0,                                 # w dot
                         0.0,                                 # phi dot
                         0.0,                                 # theta dot
                         Va/R*np.cos(gamma), # psi dot
                         0.0,                                 # p dot
                         0.0,                                 # q dot
                         0.0])                                # r dot 

    p0 = [0.0,0.0,0.0] # initial guess of alpha, beta, and phi
    x = [Va,R,gamma] # Desired Va, R, and gamma

    plsq = leastsq(residuals, p0, args=(xdotStar,x)) # Minimize funciton, returns the optimal
                                                     # alpha, beta, and phi
    print(plsq)


    alpha,beta,phi = plsq[0]
    #print('phi', phi)
    
    # Gets the trim conditions based on the optimal alpha, beta, and phi
    fx,xtrim,utrim = get_fx(alpha,beta,phi,Va,R,gamma)

    printValues(fx,xtrim,utrim,plsq)
    print('xdotstar', xdotStar)

    return xtrim,utrim


if __name__ == "__main__": 
   
    Va = float(35.0)
    gamma = float(10*np.pi/180)

    R = float(200)
    xtrim,utrim = ComputeTrim(Va,R,gamma)
    # fx,xtrim,utrim = get_fx(1,1,1,Va,R,gamma)
    # printValues(fx, xtrim, utrim,(np.array([1.0,1.0,1.0]),2))
    
