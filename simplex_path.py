import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm,trange
arr = np.asarray
norm = np.linalg.norm
uni = np.random.uniform
from opensimplex import OpenSimplex
import uuid
xaxis,yaxis,zaxis = arr([1.,0.,0.]), arr([0.,1.,0.]), arr([0.,0.,1.])

def simplex_path(c0,c1,npoints=10,start_dir=None,end_dir=None,fade=10.,feature_size=[10.],displacement=0.05,weight_mode='exponential',use_ongoing_direction=True):

    # Generates an interpolated mandering path between two points (c0 and c1) using simplex noise
    # c0: start coordinate (x,y)
    # c1: end coordinate (x,y)
    # npoints: number of points in the interpolated curve
    # start_dir: direction of line at c0, which interpolated points will align with (unit vector, (x,y))
    # end_dir: direction of line at c1, which interpolated points will align with (unit vector, (x,y))
    # fade: exponent describing the weighting of start and end directions
    # feature_size: length scale of simplex noise (list)
    # displacement: linear sclaing of simplex noise, tangential to midline

    # Separate x and y coordinates from inputs
    x = np.linspace(c0[0],c1[0],npoints)
    y = np.linspace(c0[1],c1[1],npoints)

    coords = arr([x,y]).transpose()

    length = norm(c1 - c0)
    direction = (c1 - c0) / length

    # Simplex noise seed initialisation
    seedx = uuid.uuid1().int>>64
    simplex_x = OpenSimplex(seedx)
    
    # Pre-define storage
    noise = np.zeros([coords.shape[0],2])
    points = np.zeros(coords.shape)

    # If start / end directions not supplied, use vessel direction
    if start_dir is None and use_ongoing_direction:
        start_dir = direction
    if end_dir is None and use_ongoing_direction:
        end_dir = -direction
        
    if start_dir is not None:
        # Weighting for start direction
        if weight_mode=='linear':
            start_dir_weight = np.linspace(1.,-1.,coords.shape[0])
            start_dir_weight[start_dir_weight<0.] = 0.
        elif weight_mode=='exponential':
            start_dir_weight = np.exp(-fade*np.linspace(0,length,coords.shape[0]))
        else:
            print('Warning: weight mode not recognised!')
            return None
            
        start_path = np.zeros(coords.shape)
        start_path[0] = c0 
        
        for i in range(1,coords.shape[0]):
            cur_dir = coords[i]-coords[i-1]
            cur_len = norm(coords[i]-coords[i-1])
            cur_dir /= cur_len 
            start_path[i] = start_path[i-1] + start_dir*cur_len
 
    if end_dir is not None: 
        # Weighting for end direction
        if weight_mode=='linear':        
            end_dir_weight = np.linspace(-1.,1.,coords.shape[0])
            end_dir_weight[end_dir_weight<0.] = 0.
        elif weight_mode=='exponential':    
            end_dir_weight = np.flip(np.exp(-fade*np.linspace(0,length,coords.shape[0])))

        end_path = np.zeros(coords.shape)
        end_path[-1] = c1
    
        for i in range(coords.shape[0]-1,0,-1):
            cur_dir = coords[i]-coords[i-1]
            cur_len = norm(coords[i]-coords[i-1])
            cur_dir /= cur_len 
            end_path[i-1] = end_path[i] - end_dir*cur_len

    # Define interpolated points
    points[0] = coords[0]

    # Create circular path through Simplex noise domain (i.e. that starts and ends on the same value)
    circ = arr([np.cos(np.linspace(-np.pi,np.pi,coords.shape[0])) + np.random.normal(0.,1000.) , np.sin(np.linspace(-np.pi,np.pi,coords.shape[0])) + np.random.normal(0.,1000.)  ]).transpose()
    for i in range(coords.shape[0]):
        for k,sc in enumerate(feature_size):
            noise[i,0] += simplex_x.noise2d(circ[i,0]/sc,circ[i,1]/sc) * sc
            noise[i,1] += simplex_x.noise2d((circ[i,0]/sc)+1000.,(circ[i,1]/sc)+1000.) * sc
    # Offset the initial / end noise value to zero
    noise -= noise[0]
    
    weight = np.sin(np.linspace(0,np.pi,coords.shape[0]))
    for i in range(1,coords.shape[0]):
        cur_dir = coords[i]-coords[i-1]
        cur_len = norm(coords[i]-coords[i-1])
        cur_dir /= cur_len 
        
        # Tangent to midline
        tang1 = arr([cur_dir[1],-cur_dir[0]]) 
        
        #print(start_dir_weight[i],end_dir_weight[i])     
        
        new_point = (((1.-end_dir_weight[i])+(1.-start_dir_weight[i]))*coords[i] + start_dir_weight[i]*start_path[i] + end_dir_weight[i]*end_path[i]) / 2.
            
        points[i,:] = new_point + weight[i]*tang1*noise[i,0]*length*displacement #+ weight[i]*cur_dir*noise[i,1]*length*displacement
        
    #if start_dir is not None:
    #    v = c0+start_dir*length*0.5
    #    plt.plot([c0[0],v[0]],[c0[1],v[1]],c='g')
    #if end_dir is not None:
    #    v = c1+end_dir*length*0.5
    #    plt.plot([c1[0],v[0]],[c1[1],v[1]],c='y')
        
    return points
    
def simplex_path_multi(coords,**kwargs):

    res = []
    prev_end_dir,prev_start_dir = None,None
    for i,crd in enumerate(tqdm(coords)):
        if i>0:
            if i==1 and 'start_dir' in kwargs.keys():
                start_dir = kwargs['start_dir']
            elif i!=coords.shape[0]-1:
                start_dir = coords[i+1]-coords[i-1]
                start_dir /= norm(start_dir)
            else:
                start_dir = coords[i]-coords[i-1]
                start_dir /= norm(start_dir)
            if i==coords.shape[0]-1 and 'end_dir' in kwargs.keys():
                end_dir = kwargs['end_dir']
            elif i!=coords.shape[0]-1:
                end_dir = coords[i+1]-coords[i-1]
                end_dir /= norm(end_dir)
            else:
                end_dir = coords[i]-coords[i-1]
                end_dir /= norm(end_dir)
                
            if end_dir is not None and start_dir is not None:
                #tang = arr([(x+y)/2. for x,y in zip(start_dir,end_dir)])
                #tang /= norm(tang)
                #print('Tang:',tang)
                if i!=1 and prev_end_dir is not None:
                    #start_dir = arr([(x+y)/2. for x,y in zip(start_dir,prev_end_dir)])
                    start_dir = prev_end_dir
                #if i!=coords.shape[0]:
                #    end_dir = tang
                
            kwargs_mod = {}
            for k,v in zip(kwargs.keys(),kwargs.values()):
                if k not in ['start_dir','end_dir']:
                    kwargs_mod[k] = v
              
            #print(start_dir,end_dir,prev_start_dir,prev_end_dir)      
            path = simplex_path(coords[i-1],coords[i],start_dir=start_dir,end_dir=end_dir,**kwargs_mod)
            res.append(path)
            
            prev_end_dir,prev_start_dir = end_dir,start_dir
            
    return np.concatenate(res)
