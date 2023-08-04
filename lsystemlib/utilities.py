import numpy as np
arr = np.asarray
norm = np.linalg.norm
xaxis = arr([1.,0.,0.])
yaxis = arr([0.,1.,0.])
zaxis = arr([0.,0.,1.])
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splev,splprep

# Avoiding numpy clip performance issues...
def clip_scalar(value,vmin,vmax):
    return vmin if value<vmin else vmax if value>vmax else value
def clip_array(value,vmin,vmax):
    return np.core.umath.maximum(np.core.umath.minimum(value,vmax),vmin)
uni = np.random.uniform

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_vector(x0,x1,ax=None,c='blue'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([x0[0],x1[0]],[x0[1],x1[1]],[x0[2],x1[2]],c=c)
    return ax
    
def plot_3d(coords,ax=None,c='blue'):
    """
    3D plot of coordinate array ([n,3])
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for cc in coords:
        ax.scatter(cc[0],cc[1],cc[2],c=c)
    return ax
    
def plot_cylinder(p0,p1,R,ax=None,c='blue',alpha=0.25):

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    origin = np.array([0, 0, 0])

    #vector in direction of axis
    v = p1 - p0

    #find magnitude of vector
    mag = norm(v)

    #unit vector in direction of axis
    v = v / mag

    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])

    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= norm(n1)

    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)

    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 50) # 200
    theta = np.linspace(0, 2 * np.pi, 25) # 100

    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)

    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) *       n2[i] for i in [0, 1, 2]]
    
    ax.plot_surface(X, Y, Z, color=c,alpha=alpha) #,facecolors = col,edgecolor = "red")

    #plot axis
    #ax.plot(*zip(p0, p1), color = 'red')
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 10)
    #ax.set_zlim(0, 10)
    #plt.axis('off')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    #plt.show()
    return ax
    
def normprior(x,mean=0.,sd=1.):

    v = -np.square(x-mean)/(2.*np.square(sd))
    # Clip to avoid underflow
    v = np.clip(v,-700.,700.)
    res = np.exp(v)  
    return res
    
def normalise(v):
    if norm(v)>0:
        v /= norm(v)
    return v 
    
def project_vector_onto_sphere(start_coord,end_coord,sphere_centre):

    sphere_normal = arr(start_coord - sphere_centre)
    sphere_normal /= norm(sphere_normal)
    
    direction = end_coord - start_coord
    length = norm(direction)
    
    rot_axis = cross(direction,sphere_normal)
    rot_axis /= norm(rot_axis)
    
    # Assume small displacement relative to size of sphere, so use tangent of starting point
    angle = np.arcsin(np.dot(sphere_normal,direction)/(norm(direction)*norm(sphere_normal)))
    proj_dir = geometry.rotate_about_axis(direction,rot_axis,np.rad2deg(-angle))
    proj_dir /= norm(proj_dir)
    end_coord_proj = start_coord + proj_dir
    
    return end_coord_proj,proj_dir
    
def tortured_path(points,nf=3,sd=0.1):

    nf = np.clip(nf,1,points.shape[0]-2)
    n = points.shape[0]

    f = np.linspace(0.,1.,nf+2)
    f = f[1:-1]
    
    lengths = arr([p-points[0,:] for i,p in enumerate(points)])
    length = np.sum(lengths)

    path = np.zeros([nf+2,3])

    path[0,:] = points[0,:]
    path[-1,:] = points[-1,:]
    
    yaxis = arr([0.,1.,0.])
    prev_dif = arr([0.,0.,0])
    
    def joint_pdf(x,mean0,sd0,mean1,sd1):

        pdf1 = normprior(x,mean=mean0,sd=sd0) #(1./sd0*np.sqrt(2*np.pi)) * np.exp(-0.5*(np.square(x-mean0)/sd0))
        pdf2 = normprior(x,mean=mean1,sd=sd1) #(1./sd1*np.sqrt(2*np.pi)) * np.exp(-0.5*(np.square(x-mean1)/sd1))
        pdf = pdf1 * pdf2
        pdf[np.isnan(pdf)] = 0.
        pdf = pdf / np.nansum(pdf)
        pdf[np.isnan(pdf)] = 0.
        return pdf
    
    for i in range(nf):
        ind = int(n*f[i])

        v = points[ind-1,:] - points[ind,:]
        
        # Define axes
        norm1 = np.cross(v,yaxis)
        norm1 /= norm(norm1)
        norm2 = np.cross(v,norm1)
        norm2 /= norm(norm2)
        norm3 = np.cross(norm2,norm1)
        norm3 /= norm(norm3)
        
        dist = np.min([np.linalg.norm(points[ind,:]-points[0,:]),np.linalg.norm(points[ind,:]-points[-1,:])])
        fdist = 1. #2.*dist/length

        prev_val = path[i-1,:]
        sd1 = sd*fdist
        sd2 = sd/2.
        x = np.linspace(-sd*4,sd*5,100)
        pdfx = joint_pdf(x,0.,sd1,prev_dif[0],sd2)
        pdfy = joint_pdf(x,0.,sd1,prev_dif[1],sd2)
        pdfz = joint_pdf(x,0.,sd1,prev_dif[2],sd2)
        
        if fdist>0.:
            #rnd = np.random.normal(0.,sd*fdist,3)
            rnd = [np.random.choice(x,p=pdfx),np.random.choice(x,p=pdfy),np.random.choice(x,p=pdfz)]
        else:
            rnd = np.zeros(3)
            
        rnd[1] = 0.
        path[i+1,:] = points[ind,:] + norm1*rnd[0] + norm2*rnd[1] + norm3*rnd[2]
        prev_dif = path[i+1,:]-points[ind,:]
        
        #if i==0:
        #    ax = plt.figure()
        #plt.plot(x,pdfx)
        #plt.scatter(prev_dif[0],0)
    #plt.show()
        

    try:
        from scipy.interpolate import interp1d
        from scipy import interpolate
        tck, u = interpolate.splprep([path[:,0],path[:,1],path[:,2]], s=2)
        #x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0,1,n)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        path_int = np.vstack([x_fine, y_fine, z_fine]).transpose()
        path_int[0,:] = points[0,:]
        path_int[-1,:] = points[-1,:]
    except Exception as e:
        path_int = points

    return path_int   
    
def tortuous_path(p1,p2,mag=0.9,mag2=None,smooth=True,spline_smooth=0.05,ninterp=100,plot=False,normal=None):

    # Create the list of points, and add the first two points to it.
    P = []
    P.append(p1)
    P.append(p2)

    temp = [a for a in P]

    N = 1 # The number of initial subintervals. This code currently doesn't
          # have functionality for more than one built in.

    for j in range(0,4):
        def add_point(i,mag=0.9,mag2=None):
            """
            Create a new point and add it to a temporary list of points.
            Parameters:
            -----------
            i : int
                The number of the iteration minus one
            Returns:
            --------
            None
            """

            x1,y1,z1 = P[i]
            x2,y2,z2 = P[i+1]
            length = np.linalg.norm(P[i+1]-P[i])
            direction = P[i+1] - P[i]
            direction = direction / length
            midpoint = P[i] + direction*length/2.
            
            if normal is not None:
                tangent = normal.copy()
            elif np.all(direction==zaxis):
                tangent1 = xaxis.copy()
            else:
                tangent1 = np.cross(direction,zaxis)
            tangent2 = np.cross(direction,tangent1)
            
            if mag2 is None:
                mag2 = mag
            r1 = np.random.uniform(-mag/(j+1),mag/(j+1))
            r2 = np.random.uniform(-mag2/(j+1),mag2/(j+1))
            pnew = midpoint + r1*tangent1
            pnew = pnew + r2*tangent2
            
            temp.insert(i*2+1,pnew)

        for i in range(0,len(P) - 1):
            add_point(i,mag=mag,mag2=mag2)

        P = [a for a in temp]
        
    P = np.asarray(P)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i,pnt in enumerate(P):
            if i>0:
                ax.plot([P[i-1,0],P[i,0]], [P[i-1,1],P[i,1]],[P[i-1,2],P[i,2]],c='b')

    if smooth:
        T = np.zeros(P.shape[0])
        for i,pnt in enumerate(P):
            if i>0:
                T[i-1] = np.linalg.norm(P[i]-P[i-1]) # Parameterizes the curve

        X = np.asarray(P[:,0])
        Y = np.asarray(P[:,1])
        Y = np.asarray(P[:,1])
        tck, u = splprep([P[:,0],P[:,1],P[:,2]],per=False,s=spline_smooth)
        Psmooth = splev(np.linspace(0,1,ninterp), tck)
        Psmooth = arr(Psmooth).transpose()

        if plot:
            for i,pnt in enumerate(Psmooth):
                if i>0:
                    ax.plot([Psmooth[i-1,0],Psmooth[i,0]], [Psmooth[i-1,1],Psmooth[i,1]],[Psmooth[i-1,2],Psmooth[i,2]],c='r')
        P = Psmooth
        P[0,:] = p1
        P[-1,:] = p2

    if plot:        
        plt.show()
        
    return P     
