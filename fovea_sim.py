import numpy as np
arr = np.asarray
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from vessel_sim import geometry

def gauss(x,sigma,mu,A):
    g = np.exp(-np.square(x-mu)/np.square(sigma))
    g = g*A / np.max(g)
    return g
    
def plot_fovea(coords, show=True, ax=None, normals=None):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')

    ax.scatter(coords[:,0],coords[:,1],coords[:,2],s=1)
    
    mn = np.min(coords)
    mx = np.max(coords)
    
    mn = np.min([mn,-mx])
    mx = np.max([mx,-mn])
    
    ax.set_xlim3d(mn, mx)
    ax.set_ylim3d(mn, mx)
    ax.set_zlim3d(mn, mx)
    
    if show:
        plt.show()
    return ax
    
def ensure_face_direction(face,verts,direction=1):
    ndir = np.cross(verts[1]-verts[0],verts[2]-verts[0])
    if (ndir[2]<0 and direction>=0) or (ndir[2]>0 and direction<0):
        face[[0,2]] = face[[2,0]]
    return face

def fovea_projection(n=1001,rng=[-3000,3000],rng_nsd=4,hr1=81,hr2=120,dr=100.,fw=500.,sd1=400., sd2=500.,sdl=1000.,sdr=1000.,dr1=0.,dr2=0.,flatness=0.5,
                     rhs_disp=0.,lhs_disp=0.,adaptive=True,hole=False,intersection_surface=None):
                     
    if adaptive:
        sd = np.max([sdl,sdr])
        #rng = [ -(fw/2. + sdl*rng_nsd), (fw/2. + sdr*rng_nsd) ]
        rng = [ -(fw/2. + sd*rng_nsd), (fw/2. + sd*rng_nsd) ]

    # Displacement along projection
    x = np.linspace(rng[0],rng[1],n)

    # Equilibrium position
    equ = 0.
    
    """
    # Height above equilibrium position
    hr1, hr2 = 81,120
    # Fovea depth
    dr = 100.
    # Fovea width
    fw = 500.
    # Height of dip above fovea depth
    dr1, dr2 = 0, 0
    # SD of left/right Gaussians
    sd1, sd2 = 400., 500.
    # SD of left/right roll-off Gaussians
    sdl, sdr = 1000., 1000.
    """
    
    if flatness is not None:
        min_sd = x[1]-x[0]
        if flatness==0:
            sd1,sd2 = fw/4.,fw/4.
        elif flatness==1:
            sd1,sd2 = min_sd,min_sd
        else:
            sd1 = (((fw/4.)-min_sd) * (1.-flatness)) + min_sd
            sd2 = sd1
    
    # Length of flat region at bottom
    flat_width = np.clip(fw - (sd1*2 + sd2*2),0.,None)

    a1 = hr1 + dr - dr1
    g1 = -gauss(x,sd1,-flat_width/2.,a1) + hr1
    v1 = g1.min() #g1[x>-flat_width/2.][0]
    g1[x>-flat_width/2.] = np.nan
    
    a2 = hr2 + dr - dr2
    g2 = -gauss(x,sd2,flat_width/2.,a2) + hr2
    v2 = g2.min() # g2[x>flat_width/2.][0]
    g2[x<flat_width/2.] = np.nan

    nsd = 2.
    mul, Al = -(flat_width/2.)-(sd1*nsd), hr1-equ-lhs_disp #-1.
    gl = gauss(x,sdl,mul,Al) - Al
    g1[x<mul] += gl[x<mul]

    mur, Ar = (flat_width/2.)+(sd2*nsd), hr2-equ-rhs_disp
    gr = gauss(x,sdr,mur,Ar) - Ar
    g2[x>mur] += gr[x>mur]

    #res = np.concatenate([g1[:int(n/2)],g2[int(n/2):]])
    res = np.concatenate([g1[x<=0],g2[x>0]])

    snan = np.where(np.isnan(res))
    if len(snan[0])>0 and not hole:
        #nl = len(res[((-flat_width/2)<x) & (x<(flat_width/2))])
        #res[((-flat_width/2)<x) & (x<(flat_width/2))] = np.linspace(v1,v2,nl)
        i0 = snan[0][0]-1 #np.max([0,snan[0][0]-1])
        v1 = g1[i0]
        i1 = snan[0][-1]+1 #np.min([snan[0][-1]+1,len(g2)])
        v2 = g2[i1]
        nl = len(snan[0])
        res[snan[0]] = np.linspace(v1,v2,nl)
        
    if hole:
        nsd = 2.5
        mul_hole = -(flat_width/2.)-(sd1*nsd)
        mur_hole = (flat_width/2.)+(sd2*nsd)
        res[(x>mul_hole) & (x<mur_hole)] = np.nan
        
    #print(sd1,sd2,sdl,sdr,hr1,hr2)
    
    return x,res
    
def fovea_sim(nr=10,nproj=1001,shoulder_height=[100.,100.],boundary_height=[0.,0.],shoulder_sd=[100.,100.],rng_nsd=4,plot=False,return_params=False,hole=False,intersection_surface=None,**kwargs):

    # Shoulder height
    if len(shoulder_height)==nr:
        hr1 = shoulder_height
        hr2 = np.flip(shoulder_height)
    else:
        hr1 = np.linspace(shoulder_height[0],shoulder_height[1],nr)
        hr2 = np.linspace(shoulder_height[1],shoulder_height[0],nr)
    
    # Boundary height
    if len(boundary_height)==nr:
        d1 = boundary_height
        d2 = np.flip(boundary_height)
    else:
        d1 = np.linspace(boundary_height[0],boundary_height[1],nr)
        d2 = np.linspace(boundary_height[1],boundary_height[0],nr)
    
    # Boundary SD
    if len(boundary_height)==nr:
        sds1 = boundary_height
        sds2 = np.flip(boundary_height)
    else:
        sds1 = np.linspace(shoulder_sd[0],shoulder_sd[1],nr)
        sds2 = np.linspace(shoulder_sd[1],shoulder_sd[0],nr)

    coords, faces, normals = [], [], [] #np.zeros([nr,nproj,3])
    angles = np.linspace(-np.pi/2.,np.pi/2.,nr,endpoint=False)
    nface = 0
    
    edge_verts = []
    nverts = 0
    for j,t in enumerate(angles):
        x,res = fovea_projection(hr1=hr1[j],hr2=hr2[j],n=nproj,lhs_disp=d1[j],rhs_disp=d2[j],sdl=sds1[j],sdr=sds2[j],hole=hole,intersection_surface=intersection_surface,rng_nsd=rng_nsd,**kwargs)
        
        # Calculate gradient
        gradient = np.concatenate([ [(res[1]-res[0])/(x[1]-x[0])], arr([(res[i+1]-res[i-1])/(x[i+1]-x[i-1]) for i,c in enumerate(res[1:-1])]), [(res[-1]-res[-2])/(x[-1]-x[-2])] ])
        # Calculate normals
        normal = arr([ [ -(res[i+1]-res[i-1]), (x[i+1]-x[i-1]) ] for i in range(1,res.shape[0]-1)] )
        normal = np.concatenate([arr([(-res[1]-res[0]),(x[1]-x[0])]).reshape([1,2]), normal, arr([-(res[-1]-res[-2]),(x[-1]-x[-2]) ]).reshape([1,2]) ])
 
        n = x.shape[0]
        
        res_coords = np.zeros([len(res),3])
        res_coords[:,0] = x
        res_coords[:,2] = res
        
        res_normals = np.zeros([len(res),3])
        res_normals[:,0] = normal[:,0]
        res_normals[:,2] = normal[:,1]

        Rz = np.eye(3)
        Rz[0,0],Rz[0,1] = np.cos(t), np.sin(t)
        Rz[1,0],Rz[1,1] = -np.sin(t), np.cos(t)
        
        for i,c in enumerate(res):
            coords.append(np.dot(Rz,res_coords[i]))
            normals.append(np.dot(Rz,res_normals[i]))
        edge_verts.extend([nverts,nverts+nproj-1])
        nverts += nproj           
            
        if j>0:
            for i in range(nproj-1):
                a,b = i+((j-1)*nproj), i+((j-1)*nproj)+1,
                c,d = i+(j*nproj)+1,   i+(j*nproj)
                if i>(nproj/2)-1:
                    faces.append(ensure_face_direction(arr([a,c,b]),arr([coords[a],coords[c],coords[b]])))
                    faces.append(ensure_face_direction(arr([d,c,a]),arr([coords[d],coords[c],coords[a]])))
                else:
                    faces.append(ensure_face_direction(arr([b,c,a]),arr([coords[b],coords[c],coords[a]])))
                    faces.append(ensure_face_direction(arr([a,c,d]),arr([coords[a],coords[c],coords[d]])))
            
    # Complete faces
    j = len(angles)-1
    for i in range(1,nproj-1): # range(0...)?
        a,b = i+(j*nproj),   i+(j*nproj)+1
        c,d = nproj-i-1,     nproj-i
        if i<(nproj/2):
            faces.append(arr([b,c,a]))
            faces.append(arr([a,c,d]))
            #faces.append(ensure_face_direction(arr([a,c,b]),arr([coords[a],coords[c],coords[b]])))
            #faces.append(ensure_face_direction(arr([d,c,a]),arr([coords[a],coords[c],coords[d]])))
        else:
            faces.append(ensure_face_direction(arr([b,c,a]),arr([coords[b],coords[c],coords[a]])))
            faces.append(ensure_face_direction(arr([a,c,d]),arr([coords[a],coords[c],coords[d]])))
      
    a,b,c = [1,0,len(coords)-1]
    faces.append(ensure_face_direction(arr([a,b,c]),arr([coords[a],coords[b],coords[c]])))

    i, j = 0, len(angles)-1
    a,b = i+(j*nproj),   i+(j*nproj)+1
    c,d = nproj-i-1,     nproj-i
    faces.append(ensure_face_direction(arr([b,c,a]),arr([coords[b],coords[c],coords[a]])))
    
    # Create mesh
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh() #.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    mesh.vertices = o3d.utility.Vector3dVector(coords)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    #o3d.visualization.draw_geometries([mesh,pcd])
    
    # Normalise normals
    normals = arr(normals)
    norm_len = np.linalg.norm(normals,axis=1)
    for i in range(len(normals)):
        normals[i] /= norm_len[i]
    
    coords = np.asarray(coords)     
    if plot:
        plot_fovea(coords)
    
    faces = arr(faces)
    #faces[faces==coords.shape[0]] = 0
    
    return coords, arr(faces), edge_verts, normals, mesh
    
def fovea_projection_raised(n=1001,rng=[-3000,3000],dr=100.,fw=500.,sd1=400., sd2=500.,dr1=0.,dr2=0.,flatness=0.5,
                     rhs_disp=0.,lhs_disp=0.,adaptive=True):
                     
    if adaptive:
        rng = [ -4*(fw/2.), 4*(fw/2.) ]

    # Displacement along projection
    x = np.linspace(rng[0],rng[1],n)

    # Equilibrium position
    equ = 0.
    
    """
    # Height above equilibrium position
    hr1, hr2 = 81,120
    # Fovea depth
    dr = 100.
    # Fovea width
    fw = 500.
    # Height of dip above fovea depth
    dr1, dr2 = 0, 0
    # SD of left/right Gaussians
    sd1, sd2 = 400., 500.
    # SD of left/right roll-off Gaussians
    sdl, sdr = 1000., 1000.
    """
    
    if flatness is not None:
        min_sd = x[1]-x[0]
        if flatness==0:
            sd1,sd2 = fw/4.,fw/4.
        elif flatness==1:
            sd1,sd2 = min_sd,min_sd
        else:
            sd1 = (((fw/4.)-min_sd) * (1.-flatness)) + min_sd
            sd2 = sd1
    
    # Length of flat region at top
    flat_width = np.clip(fw - (sd1*2 + sd2*2),0.,None)

    a1 = lhs_disp
    g1 = gauss(x,sd1,-flat_width/2.,dr) + a1
    v1 = g1.max() #g1[x>-flat_width/2.][0]
    
    a2 = rhs_disp
    g2 = gauss(x,sd2,flat_width/2.,dr) + a2
    v2 = g2.max() # g2[x>flat_width/2.][0]
    
    #breakpoint()

    #res = np.concatenate([g1[:int(n/2)],g2[int(n/2):]])
    res = np.concatenate([g1[x<=0],g2[x>0]])

    nl = len(res[((-flat_width/2)<x) & (x<(flat_width/2))])
    res[((-flat_width/2)<x) & (x<(flat_width/2))] = np.linspace(v1,v2,nl)
    
    #print(sd1,sd2,sdl,sdr,hr1,hr2)
    
    return x,res    
    
def fovea_sim_raised(nr=10,nproj=1001,boundary_height=[0.,0.],plot=False,return_params=False,**kwargs):

    # Boundary height
    if len(boundary_height)==nr:
        d1 = boundary_height
        d2 = np.flip(boundary_height)
    else:
        d1 = np.linspace(boundary_height[0],boundary_height[1],nr)
        d2 = np.linspace(boundary_height[1],boundary_height[0],nr)
    
    coords = [] #np.zeros([nr,nproj,3])
    angles = np.linspace(-np.pi/2.,np.pi/2.,nr,endpoint=False)

    for j,t in enumerate(angles):
        x,res = fovea_projection_raised(n=nproj,lhs_disp=d1[j],rhs_disp=d2[j],**kwargs)
        n = x.shape[0]
        
        res_coords = np.zeros([len(res),3])
        res_coords[:,0] = x
        res_coords[:,2] = res

        Rz = np.eye(3)
        Rz[0,0],Rz[0,1] = np.cos(t), np.sin(t)
        Rz[1,0],Rz[1,1] = -np.sin(t), np.cos(t)
        for i,c in enumerate(res):
            #coords[j,i] = np.dot(Rz,res_coords[i])
            coords.append(np.dot(Rz,res_coords[i]))
    
    coords = np.asarray(coords)     
    if plot:
        plot_fovea(coords)
        
    return coords
            
def align_with_normal(coords, normal):

    rt = geometry.align_vectors(arr([0.,0.,1.]),normal)
    for i,crd in enumerate(coords):
        coords[i] = np.dot(rt,crd)
            
    return coords
    
def fovea_test():

    #nr,nproj,rng = 50, 101, 10000.
    nr,nproj,rng = 50, 101, 10000.
    fw = 2000.  
    fovea_depth = 200.  
    # Sample surface around macula

    shoulder_sd = [2000.,2000.]
    flatness = 0.05
    
    #n = np.max([int(np.ceil(dsize[0]/grid_res)),100])
    # Optic disc params
    norm, uni = np.random.normal, np.random.uniform
    nr, nproj, rng_nsd = 26, 100, 3.5 # 50,101,50

    ssd = 50.
    shoulder_sd = [100.,100.] #[ssd+norm(0.,50.),ssd+norm(0.,50.)]

    fw = uni(0.7,1.2)*1000. # um
    shoulder_height = 800.
    fovea_depth = 400. #100.
        
    coords, faces, edge_verts, normals, mesh =   fovea_sim(nr=nr,nproj=nproj,shoulder_height=[100.,100.],fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=[0.,0.],shoulder_sd=shoulder_sd,plot=False)
    #coords2, faces2 = fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth*0.1,flatness=flatness,boundary_height=[-1.5*fovea_depth,-1.5*fovea_depth],plot=False)
    #coords3, faces3 = fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=[-3*fovea_depth,-3*fovea_depth],plot=False)
    
    if True:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh() #.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        mesh.vertices = o3d.utility.Vector3dVector(coords)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.visualization.draw_geometries([mesh],mesh_show_wireframe=True) #,mesh_show_back_face=True)
    
        return
    #breakpoint()
    
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(coords[:,0], coords[:,1], triangles=faces)
    fig, ax = plt.subplots(subplot_kw =dict(projection="3d"),figsize=[12,12])
    ax.plot_trisurf(coords[:,0],coords[:,1], coords[:,2], triangles=triang.triangles)
    #ax.scatter(coords[edge_verts,0],coords[edge_verts,1],coords[edge_verts,2],c='yellow',s=0.2)
    plt.show()

def on_test():

    import open3d as o3d

    #nr,nproj,rng = 50, 101, 10000.
    nr, nproj, rng_nsd = 26, 100, 3.5 # 50,101,50
    fw = [1600.,1800.,2000.]
    fovea_depth = [200.,10000.,200.]
    # Sample surface around macula
    shoulder_sd = [1000.,1000.]
    flatness = [0.1,0.9,0.1]
    
    #n = np.max([int(np.ceil(dsize[0]/grid_res)),100])
    # Optic disc params
    norm, uni = np.random.normal, np.random.uniform
    nr, nproj, rng_nsd = 26, 100, 3.5 # 50,101,50
    shoulder_height = [1000.,1000.]
    hole = [False,False,False]
    displacement = [0.,-200.,-400.]
    
    col = [ [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0],
          ]
        
    nsurf = len(fovea_depth)
    coords_store,faces_store,edge_verts_store, normals_store, meshes, crd_range = [],[],[],[],[],[]
    for i in range(nsurf):
        coords, faces, edge_verts, normals, mesh =   fovea_sim(nr=nr,nproj=nproj,shoulder_height=shoulder_height,fw=fw[i],dr=fovea_depth[i],flatness=flatness[i],boundary_height=[0.,0.],shoulder_sd=shoulder_sd,plot=False,hole=hole[i])
        coords[:,2] += displacement[i]
        coords_store.append(coords)
        faces_store.append(faces)
        edge_verts_store.append(edge_verts)
        normals_store.append(normals)
        crd_range.append([np.min(coords,axis=0),np.max(coords,axis=0)])
    crd_range = np.asarray(crd_range)

    for i in range(nsurf):
        mesh = o3d.geometry.TriangleMesh() #.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        mesh.vertices = o3d.utility.Vector3dVector(coords_store[i])
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals_store[i])
        mesh.triangles = o3d.utility.Vector3iVector(faces_store[i])
        mesh.paint_uniform_color(col[i])
        meshes.append(mesh)
        
    def delete_faces_below_target_mesh(src_mesh, target_mesh):

        scene = o3d.t.geometry.RaycastingScene() 
        target_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target_mesh))
        
        coords = np.asarray(src_mesh.vertices)
        normals = np.asarray(src_mesh.vertex_normals)
        faces = np.asarray(src_mesh.triangles)
     
        # Project rays upwards from source mesh and see where they hit target mesh
        dir_ = np.tile([0,0,1],(coords.shape[0],1))
        ray_dirs = np.vstack([coords[:,0],coords[:,1],coords[:,2],dir_[:,0],dir_[:,1],dir_[:,2]]).transpose()

        # Convert to tensors
        rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        intersections = ans['geometry_ids'].numpy().astype('int8')
        intersections[intersections==scene.INVALID_ID] = -1
        t_hit = ans['t_hit'].numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_store[src_ind][intersections>=0])
        pcd.paint_uniform_color([1,0,0])
        
        mesh = o3d.geometry.TriangleMesh() #.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        edit_coords = coords_store[src_ind][intersections>=0]
        del_inds = np.linspace(0,coords_store[src_ind].shape[0]-1,coords_store[src_ind].shape[0],dtype='int')[intersections<0]
        keep_inds = np.linspace(0,coords_store[src_ind].shape[0]-1,coords_store[src_ind].shape[0],dtype='int')[intersections>=0]
        nv = keep_inds.shape[0]
        new_inds = np.linspace(0,nv-1,nv,dtype='int')
        new_faces = arr([f for f in faces if not np.any(np.in1d(f,del_inds))])
        unq = np.unique(new_faces)
        new_faces_refac = new_faces.copy()
        for i,ind in enumerate(keep_inds):
            new_faces_refac[new_faces==ind] = new_inds[i]
        
        mesh.vertices = o3d.utility.Vector3dVector(coords[keep_inds])
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals[keep_inds])
        mesh.triangles = o3d.utility.Vector3iVector(new_faces_refac)
        
        return mesh
    
    src_ind, target_ind = 2, 1 
    mesh = delete_faces_below_target_mesh(meshes[src_ind], meshes[target_ind])
    meshes[src_ind] = mesh
    
    # Interrogate where rays from mesh 1 intersect with mesh 2
    o3d.visualization.draw_geometries(meshes,mesh_show_wireframe=True,mesh_show_back_face=True)
    breakpoint()

def hole_test():

    nr,nproj,rng = 50, 101, 10000.
    fw = 2000.  
    fovea_depth = 200.  
    # Sample surface around macula
    shoulder_sd = [1000.,1000.]
    flatness = 0.
    hole = True
    coords, faces, edge_verts =   fovea_sim(nr=nr,nproj=nproj,shoulder_height=[100.,100.],fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=[0.,0.],shoulder_sd=shoulder_sd,plot=False,hole=hole)
    #coords2, faces2 = fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth*0.1,flatness=flatness,boundary_height=[-1.5*fovea_depth,-1.5*fovea_depth],plot=False)
    #coords3, faces3 = fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=[-3*fovea_depth,-3*fovea_depth],plot=False)
    
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(coords[:,0], coords[:,1], triangles=faces)
    fig, ax = plt.subplots(subplot_kw =dict(projection="3d"))
    ax.plot_trisurf(coords[:,0],coords[:,1], coords[:,2], triangles=triang.triangles)
    ax.scatter(coords[edge_verts,0],coords[edge_verts,1],coords[edge_verts,2],c='yellow',s=0.2)
    plt.show()

if __name__=='__main__':

    if True:
        fovea_test()
    else:
        on_test()
    
    #plt.plot(coords[:nproj,2])
    
    if False:
        nrm = np.asarray([0.5,0.5,0.5])
        nrm = nrm / np.linalg.norm(nrm)
        coords = align_with_normal(coords,nrm)
            
    #ax = plot_fovea(coords,show=True)
    #plot_fovea(coords2,ax=ax,show=False)
    #plot_fovea(coords3,ax=ax)
    
