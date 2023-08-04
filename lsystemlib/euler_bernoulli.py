import matplotlib.pyplot as plt
import numpy as np

def euler_bernoulli(x, E=200e9, rho=7800, r=0.05, N=40, Nmodes=12, bcs='cantilever',loadcase='uniformpres', plot=False):

    """
    E = Young's modulus
    rho = density
    r = radius
    I = b*h**3/12 #beam moment of inertia
    N = 40 #number of elements 
    Nnodes = N+1 #number of nodes
    ell = L/N #length of beam element
    Nmodes = 12 #number of modes to keep
    mass = rho*L*A #total mass
    Irot = 1/12*mass*L**2 #rotary inertia
    x = np.arange(0,ell+L,ell)
    
    loadcase = 'uniformpres' #'pointforce' # 'pointmoment' # 'unifromdistribmoment' Input load case here
     # 'uniformpres' # 'concmomenteachnode'
    
    # Boundary conditions:
    bcs = 'cantilever' # 'clamped-clamped' # 'clamped-sliding' # simplysupported # 'cantilever' Input boundary conditions here
    """
   
    L = np.sum([np.linalg.norm(x[i]-x[i-1]) for i,_ in enumerate(x) if i>0])
    #A = b*h
    A = np.pi*r**2
    I = A**3/12
    N = x.shape[0]-1
    Nnodes = N+1 #number of nodes
    ell = L/N #length of beam element
    mass = rho*L*A #total mass
    Irot = 1/12*mass*L**2 #rotary inertia

    K = np.zeros((2*(N+1),2*(N+1))) 
    # M=zeros(2*(N+1),2*(N+1))
    for i in range(1,N+1):
        K_e = E*I[i-1]/ell**3*np.array([ [12, 6*ell, -12, 6*ell],
                                [6*ell, 4*ell**2, -6*ell, 2*ell**2],
                                [-12, -6*ell, 12, -6*ell], #x2
                                [6*ell, 2*ell**2, -6*ell, 4*ell**2]]) #theta2
        K[(2*(i-1)+1-1):(2*(i-1)+4), (2*(i-1)+1-1):(2*(i-1)+4)]=K[(2*(i-1)+1-1):(2*(i-1)+4)][:,(2*(i-1)+1-1):(2*(i-1)+4)]+K_e

    f = np.zeros((2*Nnodes,1))

    if loadcase == 'pointforce':
        #load: concentrated force on center of beam
        f[Nnodes-1] = 1 
    elif loadcase == 'pointmoment':
        #load: concentrated moment on center of beam
        f[Nnodes] = 1 
    elif loadcase == 'uniformdistribmoment':
        #load: uniform distributed moment on entire beam
        m = 1.0
        m_el = np.zeros((4,1))
        m_el[:,0] = np.array([-m,0,m,0])
        for i in range(1,N+1):
            f[2*(i-1)+1-1:2*(i-1)+4] = f[2*(i-1)+1-1:2*(i-1)+4]+m_el
    elif loadcase == 'concmomenteachnode':
        #load: concentrated moment on each node
        f[1:2*Nnodes:2]=1 
    elif loadcase == 'uniformpres':
        #load: uniform distributed load on entire beam

        for i in range(1,N+1):
            q = -rho*A[i-1]*9.81
            q_el = np.zeros((4,1))
            q_el[:,0] = np.array([q*ell/2, q*ell**2/12, q*ell/2, -q*ell**2/12])
            f[2*(i-1)+1-1:2*(i-1)+4] = f[2*(i-1)+1-1:2*(i-1)+4]+q_el
    else:
        pass

    if bcs == 'clamped-clamped':
        #clamped-clamped beam BCs
        K = np.delete(K,[2*Nnodes-2,2*Nnodes-1],0) #K[1:2,:]=[] 
        K = np.delete(K,[2*Nnodes-2,2*Nnodes-1],1) #K[:,1:2]=[] 
        f = np.delete(f,[2*Nnodes-2,2*Nnodes-1]) #f[1:2]=[] 
        K = np.delete(K,[0,1],0) #K[1:2,:]=[] 
        K = np.delete(K,[0,1],1) #K[:,1:2]=[] 
        f = np.delete(f,[0,1]) #f[1:2]=[]

    elif bcs == 'simplysupported':
        #simply supported beam BCs
        K = np.delete(K,[2*Nnodes-2],0) #K[1:2,:]=[] 
        K = np.delete(K,[2*Nnodes-2],1) #K[:,1:2]=[] 
        f = np.delete(f,[2*Nnodes-2]) #f[1:2]=[] 
        K = np.delete(K,[0],0) #K[1:2,:]=[] 
        K = np.delete(K,[0],1) #K[1:2,:]=[] 
        f = np.delete(f,[0]) #f[1:2]=[] 
     
    elif bcs == 'cantilever':
        #cantilever beam BCs
        K = np.delete(K,[0,1],0) #K[1:2,:]=[] 
        K = np.delete(K,[0,1],1) #K[:,1:2]=[] 
        f = np.delete(f,[0,1]) #f[1:2]=[] 
     
    elif bcs == 'clamped-sliding':
        #clamped-sliding beam BCs
        K = np.delete(K,[2*Nnodes-1],0) #K[1:2,:]=[] 
        K = np.delete(K,[2*Nnodes-1],1) #K[:,1:2]=[] 
        f = np.delete(f,[2*Nnodes-1]) #f[1:2]=[] 
        K = np.delete(K,[0,1],0) #K[1:2,:]=[] 
        K = np.delete(K,[0,1],1) #K[:,1:2]=[] 
        f = np.delete(f,[0,1]) #f[1:2]=[] 
    else:
        pass

    dx_vec=np.linalg.solve(K,f)

    if bcs == 'clamped-clamped':
        #clamped-clamped
        dx = np.hstack([0., dx_vec[0:2*Nnodes-5:2], 0.]) 
        dtheta = np.hstack([0., dx_vec[1:2*Nnodes-4:2], 0.])
    elif bcs == 'simplysupported':
        #simply-supported
        dx = np.hstack([0., dx_vec[1:2*Nnodes-4:2], 0.])
        dtheta = np.hstack([dx_vec[0:2*Nnodes-3:2], dx_vec[2*Nnodes-3]])
    elif bcs == 'cantilever':
        #cantilever
        dx = np.hstack([0., dx_vec[0:2*Nnodes-2:2]])
        dtheta = np.hstack([0., dx_vec[1:2*Nnodes-1:2]])
    elif bcs == 'clamped-sliding':
        #clamped-sliding beam BCs
        dx = np.hstack([0., dx_vec[0:2*Nnodes-3:2]])
        dtheta = np.hstack([0., dx_vec[1:2*Nnodes-4:2], 0.])
    else:
        pass

    if plot:
        plt.figure(1)
        #plt.subplot(211)
        plt.plot(x,dx)
        plt.ylabel('displacement [m]')
        #plt.title(['BCs: ' bcs 'Load case: ' loadcase])
        plt.show()

        #plt.subplot(212)
        #plt.plot(x,dtheta) 
        #plt.ylabel('slope [radians]')
        #plt.xlabel('X [m]')
    
    return dx
