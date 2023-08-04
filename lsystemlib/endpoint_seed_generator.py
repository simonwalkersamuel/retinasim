import numpy as np
arr = np.asarray
norm = np.linalg.norm
import os
join = os.path.join
from pymira import spatialgraph
from matplotlib import pyplot as plt
from tqdm import trange
from scipy.stats import multivariate_normal
import scipy.spatial as sptl

#def gaussian(x,mn,sd):
#    return (1./(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*np.square((x-mn)/sd))

def repulsion_function(x,y,strength=1.5,domain=arr([[-1200,1200],[-1200,1200]]),attract=False,step_length=1.):
    
    epsilon = 1e-12
    dx,dy = domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]
    
    sdx,sdy = 0.05,0.05
    mv_norm = multivariate_normal([dx/2.,dy/2.],[[sdx,0.],[0.,sdy]])
    
    pol = 1.
    if attract:
        pol = -1.
    
    dx,dy = step_length,step_length
    dx0 = mv_norm.pdf((x+dx,y)) * pol
    dx1 = mv_norm.pdf((x-dx,y)) * pol
    dy0 = mv_norm.pdf((x,y+dy)) * pol
    dy1 = mv_norm.pdf((x,y-dy)) * pol    
    grad = arr([ (dx1-dx0)/(2*dx), (dy1-dy0)/(2*dy) ])

    return mv_norm.pdf((x,y)), grad
    
#   return np.clip((1. + ((1./((x-(dx/2.))+epsilon)) + (1./((y-(dy/2.))+epsilon)))) * strength,1.,None)

def boid_seed_generator(path,plot_data=False,domain=arr([[-1200,1200],[-1200,1200]]),nf=150,nt=150,k=20,repul_strength=0.005,edge_repulsion=True,edge_repul_strength=0.001,ntype=2):
    
    density = nf/(np.pi*np.square((domain[0,1]-domain[0,0])/2.))
    print('Density estimate: {} boids per square unit'.format(density))
    
    step_length = 0.001
    epsilon = 1e-12

    coords = np.zeros([nt,nf,2])

    # Random initialisation
    for i in range(nf):
        coords[0,i] = np.random.uniform(domain[0,0],domain[0,1]),np.random.uniform(domain[1,0],domain[1,1])

    nf_nxt = nf
    
    def plot(coords,show=False,filename=None):
        plt.clf()
        npts = int(coords.shape[0] / float(ntype))
        plt.scatter(coords[:npts,0],coords[:npts,1],c='blue',s=4)
        plt.scatter(coords[npts:,0],coords[npts:,1],c='red',s=4)
        
        dsize = [domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]]
        plt.xlim([domain[0,0]-dsize[0]*0.5, domain[0,1]+dsize[0]*0.5])
        plt.ylim([domain[1,0]-dsize[1]*0.5, domain[1,1]+dsize[1]*0.5])
        
        plt.plot([domain[0,0],domain[0,0]],[domain[1,0],domain[1,1]],c='gray')
        plt.plot([domain[0,1],domain[0,1]],[domain[1,0],domain[1,1]],c='gray')
        plt.plot([domain[0,0],domain[0,1]],[domain[1,0],domain[1,0]],c='gray')
        plt.plot([domain[0,0],domain[0,1]],[domain[1,1],domain[1,1]],c='gray')
        
        ax = plt.gca()
        #ax.axis('equal')
        ax.set_aspect('equal', 'box')

        if show:
            plt.show()
        elif filename is not None:
            plt.savefig(filename)
            
    plot(coords[0],filename=join(path,'plot_{}'.format(0)))

    for t in trange(1,nt):
        #print('t={}, nf={}'.format(t,nf))

        for f in range(nf):
            dists = arr([norm(coords[t-1,f]-x) for i,x in enumerate(coords[t-1])])

            if False:
                idx = np.argpartition(dists, k)
                rnd_inds = idx[:k+1]
                rnd_inds = rnd_inds[rnd_inds!=f]
            else:
                rnd_inds = np.linspace(0,nf-1,nf,dtype='int')
                rnd_inds = np.delete(rnd_inds,np.where(rnd_inds==f))
            
            # Average position
            loc_coords = coords[t-1,rnd_inds]
            loc_com = np.mean(loc_coords,axis=0)
            loc_direction = loc_com - coords[t-1,f]
            com_dist = norm(loc_com - coords[t-1,f])
            loc_direction = loc_direction*step_length / (com_dist) #*float(nf))
            
            # Average heading
            if False:
                loc_coords0 = coords[t-2,rnd_inds]
                loc_coords1 = coords[t-1,rnd_inds]
                loc_heading = np.mean(loc_coords1 - loc_coords0,axis=0)
                loc_heading = loc_heading*step_length / norm(loc_heading)
            
            # Repulsion between particles
            repul_dir = arr([coords[t-1,f]-lc for lc in loc_coords])
            loc_dist = dists[rnd_inds]    
            
            if False: # macula region
                radius = 0.2        
                
                # Attractor
                centre = arr([domain[0,1]+domain[0,0], domain[1,1]+domain[1,0]])/2.
                vec = centre - coords[t-1,f]
                centre_dist = np.clip(norm(vec),epsilon,None)
                domain_size = np.min([domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]])/2.
                if centre_dist<domain_size*0.2:
                    repul_strength_cur = repul_strength * 100.
                elif centre_dist<domain_size*0.5:
                    repul_strength_cur = repul_strength * 0.2
                else:
                    repul_strength_cur = repul_strength
                    
                # Repul
                #centre = arr([domain[0,1]+domain[0,0], domain[1,1]+domain[1,0]])/2.
                #vec = centre - coords[t-1,f]
                #centre_dist = np.clip(norm(vec),epsilon,None)
                #domain_size = np.min([domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]])/2.
                if centre_dist<domain_size*0.2:
                    repul_strength_cur = repul_strength * 100.
                else:
                    repul_strength_cur = repul_strength
            else:
                repul_strength_cur = repul_strength
            repul_dir = arr([x*repul_strength_cur/(loc_dist[k]*norm(x)) for k,x in enumerate(repul_dir)])
            repul_dir = np.mean(repul_dir,axis=0)
            
            # Edge repulsion
            if edge_repulsion:
                edge_mode = 'circle'
                if edge_mode=='rectangle':
                    dirx0,diry0 = domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]
                    dirx0,diry0 = dirx0 / norm(dirx0), diry0 / norm(diry0)
                    dirx1,diry1 = -dirx0, -diry0
                    dx0 = coords[t-1,f,0] - domain[0,0]
                    dx1 = domain[0,1] - coords[t-1,f,0]
                    dy0 = coords[t-1,f,1] - domain[1,0]
                    dy1 = domain[1,1] - coords[t-1,f,1]
                    ex0 = dirx0 / np.square(dx0+epsilon)
                    ex1 = dirx1 / np.square(dx1+epsilon)
                    ey0 = diry0 / np.square(dy0+epsilon)
                    ey1 = diry1 / np.square(dy1+epsilon)
                    edge_repul = (ex0 + ex1 + ey0 + ey1) * edge_repul_strength
                elif edge_mode=='circle':
                    radius = np.min([domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]]) 
                    centre = arr([domain[0,1]+domain[0,0], domain[1,1]+domain[1,0]])/2.
                    vec = centre - coords[t-1,f]
                    edge_dist = np.clip(radius - norm(vec),epsilon,None)
                    edge_repul = (vec/norm(vec)) * (edge_repul_strength/np.power(edge_dist,3))
            else:
                edge_repul = arr([0.,0.])
                
            # Environment repulsion / attraction
            rep,grad = arr(repulsion_function(coords[t-1,f,0],coords[t-1,f,1],domain=domain,attract=True,step_length=step_length))
            env_repul_dir = grad

            coords[t,f] = coords[t-1,f] + edge_repul + repul_dir #+ 7.5*loc_direction #+ loc_heading/2. #+ env_repul_dir 
            #coords[t,f,0] = np.clip(coords[t,f,0],domain[0,0],domain[0,1])
            #coords[t,f,1] = np.clip(coords[t,f,1],domain[1,0],domain[1,1])
                
        nf = nf_nxt
        
        if plot_data:
            plot(coords[t],filename=join(path,'plot_{}'.format(t)))
        
        #print(np.abs(np.sum(coords[t]-coords[t-1])))
        if np.abs(np.sum(coords[t]-coords[t-1]))<1e-12:
            break

    print('Complete')
    if plot_data:
        plot(coords[-1],show=True)

    return coords[-1]
    
def voronoi_network(path,base_pts):
    vor = sptl.Voronoi(points=base_pts)

    print('Extracting edges...')
    edges = [[], []]
    for facet in vor.ridge_vertices:
        # Create a closed cycle of vertices that define the facet
        edges[0].extend(facet[:-1]+[facet[-1]])
        edges[1].extend(facet[1:]+[facet[0]])
        
    print('Processing edges...')
    edges = np.vstack(edges).T  # Convert to scipy-friendly format
    mask = np.any(edges == -1, axis=1)  # Identify edges at infinity
    edges = edges[~mask]  # Remove edges at infinity
    edges = np.sort(edges, axis=1)  # Move all points to upper triangle
    # Remove duplicate pairs
    edges = edges[:, 0] + 1j*edges[:, 1]  # Convert to imaginary
    edges = np.unique(edges)  # Remove duplicates
    edges = np.vstack((np.real(edges), np.imag(edges))).T  # Back to real
    edges = np.array(edges, dtype=int)

    def inside_domain(x):
        if x[0]>=domain[0,0] and x[0]<domain[0,1] and \
           x[1]>=domain[1,0] and x[1]<domain[1,1]:
            return True
        else:
            return False

    if False: # Filter vertices outside domain       
        verts = arr([(i,x) for i,x in enumerate(vor.vertices) if inside_domain(x)]).transpose()
        vert_inds = verts[0]
        verts = verts[1]
        nverts = verts.shape[0]
        edges = arr([x for x in edges if x[0] in vert_inds and x[1] in vert_inds])
        for i,v in enumerate(vert_inds):
            edges[edges==v] = i
        vert_inds = np.arange(0,nverts,dtype='int')
    else:
        verts = vor.vertices
        nverts = verts.shape[0]
        vert_inds = np.arange(0,nverts,dtype='int')

    #def plot_voronoi_3d(base_pts,vor):
    if False:
        print('Plotting...')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #for pt in node_coords:
        #    ax.scatter(pt[0],pt[2],c='b')
        #for i,pt in enumerate(base_pts):
        #    ax.scatter(pt[0],pt[1],c='r',s=4)
        #for ax in vor.vertices:
        #    if inside_domain(pt):
        #        ax.scatter(pt[0],pt[1],c='g')
        for e in edges:
            x0 = vor.vertices[e[0],:]
            x1 = vor.vertices[e[1],:]
            if inside_domain(x0) and inside_domain(x1):
                ax.plot([x0[0],x1[0]],[x0[1],x1[1]] ,c='b')  
                
        dsize = [domain[0,1]-domain[0,0], domain[1,1]-domain[1,0]]
        ax.set_xlim([domain[0,0]-dsize[0]*0.5, domain[0,1]+dsize[0]*0.5])
        ax.set_ylim([domain[1,0]-dsize[1]*0.5, domain[1,1]+dsize[1]*0.5])
        
        ax.plot([domain[0,0],domain[0,0]],[domain[1,0],domain[1,1]],c='gray')
        ax.plot([domain[0,1],domain[0,1]],[domain[1,0],domain[1,1]],c='gray')
        ax.plot([domain[0,0],domain[0,1]],[domain[1,0],domain[1,0]],c='gray')
        ax.plot([domain[0,0],domain[0,1]],[domain[1,1],domain[1,1]],c='gray')            

        #plt.show()
        filename = join(path,'plot_final')
        plt.savefig(filename)  
