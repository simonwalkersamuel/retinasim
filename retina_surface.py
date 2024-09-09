from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
arr = np.asarray
arr = np.asarray
norm, uni = np.random.normal, np.random.uniform
from opensimplex import OpenSimplex
import uuid
import open3d as o3d
#import matplotlib.path as mpltPath
from scipy.spatial import ConvexHull, convex_hull_plot_2d        
import shapely.geometry
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
from tqdm import trange, tqdm
import os
join = os.path.join
import open3d as o3d

from retinasim import fovea_sim, geometry

"""
Class for creating retinal layer meshes
"""

class RetinaSurface():

    def __init__(self, domain=None, eye=None):
    
        if eye is None:
            from retina_sim.eye import Eye
            self.eye = Eye()
        else:
            self.eye = eye
    
        if domain is None:
            ext = 6000 #5000
            domain = arr([ [self.eye.fovea_centre[0]-1000,self.eye.fovea_centre[0]+1000.],
                                [self.eye.fovea_centre[1]-1000,self.eye.fovea_centre[1]+1000.],
                                [-ext,ext]])

        self.set_domain(domain)
        self.set_layers()
        
    def set_layers(self, add_noise=False):
        
        """
        Layers:
        ILM: Internal Limiting Membrane
        RNFL: Retinal Nerve Fibre Layer (33+/-15 um) (thickest at optic nerve, thins towards fovea
        GCL: Ganglion Cell Layer 
        IPL: Inner Plexiform Layer (GCL+IPL, 69+/-22um)
        INL: Inner Nuclear Layer (27+/-8um) (flat into fovea)
        OPL: Outer Plexiform Layer (33+/-12um)
        ONL: Outer Nuclear Layer (ONL+PIS: 77+/-21um PIS=photoreceptor inner segments)
        ELM: External Limiting Membrane (ELM1 and ELM2 are the bright layer and dark layer underneath, respectively)
        PR: Photoreceptor Layers (PIS/POS)
        RPE: Retinal Pigment Epithelium (RPE+BM: 25+/-4um)
        BM: Bruch's Membrane
        CC: Choriocapillaris
        CS: Choroidal Stroma (choroid, total thickness 250-350 um)
        CS0: Furthest extent of the choroid
        
        Thicknesses: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2590782/#:~:text=The%20thickness%20profiles%20of%20the,ranged%20between%207%20%E2%80%93%2040%20microns.
        REP thickness: https://www.nature.com/articles/s41433-021-01911-5
        """
        
        test = False
        if test:
            layers =          [ 'ILM', 'GCL', 'vessels', 'INL', 'CS' ]
            layer_thickness = [  2.,   35.,   0.,        27.,   300. ]
            seg_val =         [  1,    3,     -1,        5,     12   ]
            grey_val =        [  1.,   0.5,   -1.,       0.3,   0.5  ]
        else:
            layers =          [ 'ILM', 'RNFL', 'GCL', 'vessels', 'IPL','INL', 'OPL', 'ONL', 'ELM1', 'ELM2', 'PR1', 'PR2', 'CS',  'CS0'  ]
            layer_thickness = [  2.,    33.,    35.,  0.,        35.,  27.,   33.,   77.,   2.,     4.,     10.,   10.,   350.,  0.     ]
            seg_val =         [  1,     2,      3,    -1,        4,    5,     6,     7,     8,      9,      10,    11,    12,    13     ]
            grey_val =        [  1.,    0.9,    0.5,  0.,       0.55, 0.3,   0.5,   0.1,   0.9,    0.1,    1.,    1.,    0.5,   0.     ]
        
        self.layers = layers
        self.grey_val = grey_val
        self.seg_val = seg_val
        self.layer_thickness = arr(layer_thickness)
        
        if add_noise:
            self.layer_thickness = arr([x+norm(0.,x*0.1) for x in self.layer_thickness])
        
        self.nlayer = len(layers)
        self.nsurface = self.nlayer + 1
        self.retina_thickness = np.sum(self.layer_thickness[:self.nlayer-1])
        self.choroid_thickness = self.layer_thickness[-2]   
        
    def set_domain(self,domain):
        self.domain = domain         
        
    def surface_index_to_coord(self, inds, dims=None):
     
        domain = self.domain[0:2]
        if dims is None:
            dims = self.surface.shape.astype('float')
        else:
            dims = dims.astype('float')
        norm_coord = inds / dims
        return norm_coord*(domain[:,1]-domain[:,0]) + domain[:,0]
        
    def coord_to_surface_index(self, coords):
     
        domain = self.domain
        norm_coord = (coords - domain[:,0]) / (domain[:,1]-domain[:,0])
        return (norm_coord[0:2] * self.surface.shape).astype('int')
        
    def get_nearest_surface_normal(self,coords,surface_normals=None):
    
        x,y = self.coord_to_surface_index(coords)
        x = np.clip(x,0,self.surface.shape[0]-1)
        y = np.clip(y,0,self.surface.shape[1]-1)
        if surface_normals is None:
            return self.surface_normals[x,y]
        else:
            return surface_normals[x,y]
            
    def plot_surface(self, xv, yv, surface, normals=None, ax=None, show=True, vl=200.,skip=10,plot_surface=True):
    
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
        
        if plot_surface:    
            ax.plot_surface(xv, yv, surface,cmap=cm.coolwarm)
        
        if normals is not None:
            for i,x in enumerate(xv[:,0]):
                for j,y in enumerate(yv[0,:]):
                    if i%skip==0 and j%skip==0:
                        ax.plot([x,x+normals[i,j,0]*vl], [y,y+normals[i,j,1]*vl], [surface[i,j],surface[i,j]+normals[i,j,2]*vl])
        if show:
            plt.show()
        return ax        
        
    def create_raised_indent_surface(self,normal=arr([0.,0.,1.]),fovea_centre=arr([0.,0.,0.]),nr=20,nproj=1001,rng_nsd=4,fw=50.,shoulder_height=[100.,100.],shoulder_sd=[100.,100.],boundary_height=[0.,0.],flatness=0.2,fovea_depth=220.,raised=False,hole=False,intersection_surface=None):
    
        if not raised:
            coords, faces, edge_verts, normals, mesh = fovea_sim.fovea_sim(nr=nr,nproj=nproj,shoulder_height=shoulder_height,fw=fw,rng_nsd=rng_nsd,dr=fovea_depth,flatness=flatness,boundary_height=boundary_height,shoulder_sd=shoulder_sd,plot=False,hole=hole,intersection_surface=intersection_surface)
            
            #device = o3d.core.Device("CPU:0")
            #dtype_f = o3d.core.Dtype.Float32
            #dtype_i = o3d.core.Dtype.Int32 
            #mesh = o3d.t.geometry.TriangleMesh(device)
            #mesh.vertices["positions"] = o3d.core.Tensor(coords, dtype_f, device)
            #mesh.triangles["indices"] = o3d.core.Tensor(faces, dtype_i, device)
            #scene = o3d.t.geometry.RaycastingScene()
            #surface_id = scene.add_triangles(mesh)
            #breakpoint()
        else:
            coords = fovea_sim.fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=boundary_height,plot=False)
        
        return coords, faces, edge_verts, normals, mesh
        
    def calculate_normals(self,surface,dx=1.,dy=1.):
    
        nx,ny = surface.shape[0],surface.shape[1]
        
        normals = np.zeros([nx,ny,3])
        
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                ndy = arr([0.,dy,surface[i,j+1] - surface[i,j-1]])
                ndx = arr([dx,0.,surface[i+1,j] - surface[i-1,j]])
                normals[i,j,:] = np.cross(ndx,ndy)
                normals[i,j,:] = normals[i,j,:] / np.linalg.norm(normals[i,j,:])
                if i==1:
                    normals[0,j,:] = normals[i,j,:]
                elif i==nx-2:
                    normals[-1,j,:] = normals[i,j,:]   
                if j==1:
                    normals[i,0,:] = normals[i,j,:]
                elif j==ny-2:
                    normals[i,-1,:] = normals[i,j,:]
        return normals
        
    def nearest_surface_coord(self, surface_coords, surface_faces,surface_normals, x0):
    
        scene = o3d.t.geometry.RaycastingScene()
        start_point = o3d.core.Tensor([x0], dtype=o3d.core.Dtype.Float32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(surface_coords)
        mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals)
        mesh.triangles = o3d.utility.Vector3iVector(surface_faces)
        #mesh_ids[scene.add_triangles(mesh)] = 'mesh'
        mesh_id = scene.add_triangles(mesh)
        
        x0_snap = scene.compute_closest_points(start_point)['points'].numpy()
        
    def step(self,surface_coords,surface_faces,surface_normals,x0,dx=1.,dy=1.):
        scene = o3d.t.geometry.RaycastingScene()
        start_point = o3d.core.Tensor([x0], dtype=o3d.core.Dtype.Float32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(surface_coords)
        mesh.triangles = o3d.utility.Vector3iVector(surface_faces)
        mesh_ids[scene.add_triangles(mesh)] = 'mesh'
        
        x0_snap = scene.compute_closest_points(start_point)['points'].numpy()
        print('The closest point on the surface is', x0_snap)
        #print('The closest point belongs to triangle', x0_snap['primitive_ids'][0].item())
        
        x1 = x0_snap + arr([dx,dy,0.])
        end_point = o3d.core.Tensor([x1], dtype=o3d.core.Dtype.Float32)
        x1_snap = scene.compute_closest_points(end_point)['points'].numpy()
        return x1_snap
        
    def ensure_face_direction(self,face,verts,direction=1):
        ndir = np.cross(verts[1]-verts[0],verts[2]-verts[0])
        if (ndir[2]<=0 and direction>=0) or (ndir[2]>0 and direction<0):
            face[[0,2]] = face[[2,0]]
            tst_dir = np.cross(verts[1]-verts[2],verts[0]-verts[2])
            if (tst_dir[2]<0 and direction>=0) or (tst_dir[2]>0 and direction<0):
                #breakpoint()
                print('Weird thing happened (ensure_face_direction)')
        return face
        
    def delete_faces_containing_vertices(self,faces,del_inds,npoints):
        # Delete faces and offset vertex index references
        
        # Find faces containing deleted vertex indices, from supplied list of faces
        #del_faces = arr([bool(set(f) & set(del_inds)) for f in faces])
        del_faces = np.where((np.in1d(faces[:,0],del_inds)) | (np.in1d(faces[:,1],del_inds)) | (np.in1d(faces[:,2],del_inds)))[0]
        keep_face = np.ones(faces.shape[0],dtype='bool')
        keep_face[del_faces] = False

        # Find which vertices are affected (i.e. at the edge of deleted vertices)
        affected = faces[del_faces]
        affected_inds = np.unique(affected.reshape(affected.size))
        # Create a list of face indices that are adjacent to deleted faces
        edge_inds = arr([x for x in affected_inds if x not in del_inds])
        
        # Keep only non-deleted faces
        faces_st = faces.copy()
        faces = faces[keep_face]
        
        # Re-factor vertex references to account for vertices being deleted
        new_inds_lookup = np.zeros(npoints,dtype='int')-1
        old_inds = np.linspace(0,npoints-1,npoints,dtype='int')
        new_inds_lookup[~np.in1d(old_inds,del_inds)] = np.linspace(0,npoints-del_inds.shape[0]-1,npoints-del_inds.shape[0])
        faces_refact = new_inds_lookup[faces] 
        edge_inds_refact = new_inds_lookup[edge_inds] 
        
        # END TEST

        #count = 0
        #faces_refact = faces.copy()
        #edge_inds_refact = edge_inds.copy()
        #for ind in trange(npoints):
        #    if ind not in del_inds:
        #        inds = np.where(faces==ind)
        #        faces_refact[inds] = count
        #        inds = np.where(edge_inds==ind)
        #        edge_inds_refact[inds[0]] = count
        #        count +=1 
        #        
        #breakpoint()
                
        return faces_refact, arr(edge_inds_refact)
        
    def stitch_edges(self,edge1,edge2, plot=False, sort=False):
        faces = []
        ne1, ne2 = edge1.shape[0],edge2.shape[0]
        
        if sort:
            edge1_cw = self.sort_points_clockwise(edge1)
            edge2_cw = self.sort_points_clockwise(edge2)
        else:
            edge1_cw = edge1.copy()
            edge2_cw = edge2.copy()
        
        combined_points = np.concatenate([edge1_cw,edge2_cw])
        
        tri = Delaunay(combined_points[:,:2]) # take the first two dimensions
        faces = tri.simplices
        
        edge1_2d = np.concatenate([edge1_cw[:,0:2],edge1_cw[0,0:2].reshape([1,2])])
        edge1_poly = shapely.geometry.Polygon(edge1_2d)

        del_face = []
        #print('Stitching...')
        for f,face in enumerate(faces):
            a = combined_points[face[0],0:2]
            b = combined_points[face[1],0:2]
            c = combined_points[face[2],0:2]
            
            def midpoint(x0,x1):
                v = arr(x1) - arr(x0)
                l = np.linalg.norm(v)
                v /= l
                return arr(x0) + v*l/2.
            
            ab_midpoint = shapely.geometry.Point(midpoint(a,b))
            bc_midpoint = shapely.geometry.Point(midpoint(b,c))
            ca_midpoint = shapely.geometry.Point(midpoint(c,a))
            
            ab_int = ab_midpoint.within(edge1_poly)
            bc_int = bc_midpoint.within(edge1_poly)
            ca_int = ca_midpoint.within(edge1_poly)
            
            def are_edge1_neighbours(a,b,ne1):
                if a>=ne1 or b>=ne1:
                    return False
                if np.abs(a-b)==1:
                    return True
                if a==0 and b==ne1-1:
                    return True
                if b==0 and a==ne1-1:
                    return True
                return False
            
            ab_neigh = are_edge1_neighbours(face[0],face[1],ne1)
            bc_neigh = are_edge1_neighbours(face[1],face[2],ne1)
            ca_neigh = are_edge1_neighbours(face[2],face[0],ne1)
            
            if (ab_int and not ab_neigh) or (bc_int and not bc_neigh) or (ca_int and not ca_neigh):
                #print(face,ab_int,bc_int,ca_int,ab_neigh,bc_neigh,ca_neigh,ne1)
                del_face.append(True)
            else:
                del_face.append(False)
        
        all_faces = arr(faces.copy())
        rem_faces = arr([self.ensure_face_direction(arr([f[0],f[1],f[2]]), combined_points[[f[0],f[1],f[2]]]) for f in faces[arr(del_face)]]) # faces[arr(del_face)] # 
        faces = arr([self.ensure_face_direction(arr([f[0],f[1],f[2]]), combined_points[[f[0],f[1],f[2]]]) for f in faces[~arr(del_face)]]) # faces[~arr(del_face)] #
        
        if plot:
            points = combined_points.copy()
            
            pcd = o3d.geometry.PointCloud()
            pcd.paint_uniform_color([1, 0, 0])
            pcd.points = o3d.utility.Vector3dVector(edge1_cw)
            
            pcd2 = o3d.geometry.PointCloud()
            pcd2.paint_uniform_color([0, 0, 1])
            pcd2.points = o3d.utility.Vector3dVector(edge2_cw)
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(faces) #[~arr(del_face)])
            mesh.paint_uniform_color([0, 1, 0])
            
            meshd = o3d.geometry.TriangleMesh()
            meshd.vertices = o3d.utility.Vector3dVector(points)
            meshd.triangles = o3d.utility.Vector3iVector(rem_faces)
            
            #rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
            o3d.visualization.draw_geometries([pcd,pcd2,mesh],mesh_show_wireframe=True)  #,epcd])
        
        return faces
        
    def simulate_macula(self,layers=None,layer_thickness=None,ssd=norm(500.,10.),fw=norm(250.,50.),shoulder_sd=None,flatness=uni(0.01,0.08),gap=norm(2.,0.1),boundary_height=arr([0.,0.]),nproj=1001,nr=50,rng_nsd=4,plot=False,plot_projection=False,add_noise=True):

        print('Simulating macular geometry...')
        
        """
        Requires INL as a minimum
        """
        
        if layers is None:
            layers = self.layers
            nlayer = len(layers)
            nsurface = nlayer + 1
            layer_thickness = self.layer_thickness

        # Define geometry
        if shoulder_sd is None:
            if add_noise:
                shoulder_sd = arr([ssd+norm(0.,10.),ssd+norm(0.,10.)])
            else:
                shoulder_sd = arr([ssd,ssd])
        dip_layer = layers.index('INL')
        height = np.sum(layer_thickness[0:5])
        depth = height - np.sum(layer_thickness[3:5])
        sheight = depth #norm(depth*0.5,depth*0.05)
        shoulder_height = arr([sheight,sheight])
        
        fovea_proj = []
        
        last_height, last_width = None, None
        
        sh_scale = np.ones(nsurface)
        shoulder_decay = np.random.normal(0.8,0.1)
        for i in range(2,nsurface):
            sh_scale[i] = sh_scale[i-1] * shoulder_decay
        d_scale = np.ones(nsurface)
        d_scale[5:] = sh_scale[5:]
        
        surface_coords = [[] for _ in range(nsurface)]
        surface_normals = [[] for _ in range(nsurface)]
        surface_faces = [[] for _ in range(nsurface)]
        macula_edge = [[] for _ in range(nsurface)]
        
        asymmetric = False
        
        for i in trange(nsurface):
            
            if last_height is not None and last_height<0.:
                fwc = last_width * 0.95
            else:
                fwc = fw
            fdc = (depth - np.sum(layer_thickness[0:i]) + i*gap) * d_scale[i]
            shc = shoulder_height * sh_scale[i]
            
            if i>=layers.index('CS'):
                fdc = 0.
                shc = arr([0.,0.])
                
            # Make right-hand shoulder (i.e. closest to macula) higher and wider
            shoulder_sd_c = shoulder_sd.copy()
            if i<layers.index('GCL') and asymmetric:
                shoulder_sd_c[1] *= 2
                shc[1] *= 1.2 # np.clip(norm(1.2,0.05),1.01,None)
                shc[0] /= 1.2 #np.clip(norm(1.2,0.05),1.01,None)

            # Align with normal
            nrm = self.eye.occular_centre - self.eye.fovea_centre
            nrm = nrm / np.linalg.norm(nrm)
            
            # Translate to fovea centre
            coords, faces, edge_verts, normals, mesh = fovea_sim.fovea_sim(nr=nr,nproj=nproj,shoulder_height=shc,fw=fwc,rng_nsd=rng_nsd,dr=fdc,flatness=flatness,boundary_height=boundary_height,shoulder_sd=shoulder_sd_c,plot=False,hole=False)
            #coords, faces, edge_verts, normals, mesh = self.create_raised_indent_surface(normal=None,fovea_centre=None,nr=nr,nproj=nproj,rng_nsd=rng_nsd,fw=fwc,shoulder_height=shc,shoulder_sd=shoulder_sd,boundary_height=boundary_height,flatness=flatness,fovea_depth=fdc,raised=raised)
            fovea_proj.append(coords[:nproj,2])

            # Displace
            if i==0:
                thk = 0
            else:
                thk = np.sum(layer_thickness[:i])
            
            #displacement = arr([np.linalg.norm([xp[0]-self.eye.occular_centre[0],xp[1]-self.eye.occular_centre[1],self.eye.occular_centre[2]]) for xp in coords]) - self.eye.occular_radius - thk
            displacement = thk
            coords[:,2] -= displacement
            # Rotate around centre
            #coords = fovea_sim.align_with_normal(coords,arr([nrm[0],nrm[1],nrm[2]]))
            # Displace
            coords[:,0:2] += [self.eye.fovea_centre[0],self.eye.fovea_centre[1]]

            last_height = fovea_proj[-1][0] - fovea_proj[-1][int(nproj/2)]
            last_width = fwc
            last_shoulder_height = shc
            
            macula_edge[i] = coords[edge_verts]

            surface_coords[i] = coords
            surface_normals[i] = normals
            surface_faces[i] = faces
              
        if plot:
            meshes = []
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                meshes.append(mesh)
            
            o3d.visualization.draw_geometries(meshes,mesh_show_back_face=True,mesh_show_wireframe=True)
            #breakpoint()   
        
        if plot_projection:
            mn,mx = coords.min(axis=0),coords.max(axis=0)
            cols = np.zeros([nsurface,3],dtype='int16')
            cols[:,0] = 1
            gcl_ind = layers.index('GCL')
            cols[gcl_ind,:] = [0,0,1]
            v_ind = layers.index('GCL')
            cols[v_ind,:] = [0,1,0]        
            z0 = -1e6
            size = 1024
            
            fig, axs = plt.subplots(2,figsize=[12,12])
            no_line = True
            # X-axis
            meshes = self.projection(arr([mn[0],self.eye.fovea_centre[1],z0]),arr([mx[0],self.eye.fovea_centre[1],z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size,grey_val=self.grey_val,no_line=no_line,axis=axs[0],show=False)
            meshes = self.projection(arr([self.eye.fovea_centre[0],mn[1],z0]),arr([self.eye.fovea_centre[0],mx[1],z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size,grey_val=self.grey_val,no_line=no_line,axis=axs[1],show=False)
            
            axs[0].set(xlabel='x')
            axs[0].set_xlim([self.eye.fovea_centre[0]-1000,self.eye.fovea_centre[0]+1000])
            axs[1].set(xlabel='y')
            axs[1].set_xlim([self.eye.fovea_centre[1]-1000,self.eye.fovea_centre[1]+1000])
            
            plt.show()
            
        return surface_coords, surface_normals, surface_faces, macula_edge
        
    def delete_faces_below_target_mesh(self, src_mesh, target_mesh, edge_verts=None, max_displacement=2000.):

        scene = o3d.t.geometry.RaycastingScene() 
        target_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target_mesh))
        
        coords = np.asarray(src_mesh.vertices)
        normals = np.asarray(src_mesh.vertex_normals)
        faces = np.asarray(src_mesh.triangles)
        
        com = np.mean(coords,axis=0)
     
        # Project rays upwards from source mesh and see where they hit target mesh
        dir_ = np.tile([0,0,1],(coords.shape[0],1))
        ray_dirs = np.vstack([coords[:,0],coords[:,1],coords[:,2],dir_[:,0],dir_[:,1],dir_[:,2]]).transpose()
        rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        intersections = ans['geometry_ids'].numpy().astype('int8')
        intersections[intersections==scene.INVALID_ID] = -1
        
        # Repeat in opposite direction, to remove rays that don't intersect because outside theboundary of the target mesh (i.e. at the edge)
        #dir_ = np.tile([0,0,-1],(coords.shape[0],1))
        #ray_dirs = np.vstack([coords[:,0],coords[:,1],coords[:,2],dir_[:,0],dir_[:,1],dir_[:,2]]).transpose()
        #rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
        #ans = scene.cast_rays(rays)
        #intersections_rev = ans['geometry_ids'].numpy().astype('int8')
        #intersections_rev[intersections_rev==scene.INVALID_ID] = -1
        
        #edit_coords = coords[intersections>=0]
        disp = np.linalg.norm(coords-com,axis=1)
        all_inds = np.linspace(0,coords.shape[0]-1,coords.shape[0],dtype='int')
        dels = (intersections<0) & (disp<max_displacement)
        del_inds = all_inds[dels]
        keep_inds = all_inds[~dels]
        
        # Keep edge vertices
        if edge_verts is not None:
            sind = arr([i for i in del_inds if i in edge_verts])
            if len(sind)>0:
                #breakpoint()
                keep_inds = np.sort(np.concatenate([keep_inds,sind]))
                del_inds = arr([i for i in del_inds if i not in edge_verts])
                
        #breakpoint()
        # Find where source edge hits target
        new_vert, new_face = [], []
        count = keep_inds.shape[0]
        shift_coords = coords.copy()
        shifted = np.zeros(coords.shape[0],dtype='int')
        for face in faces:
            # Find faces that straddle edge of deleted area
            if np.any(np.in1d(face,del_inds)) and np.any(np.in1d(face,keep_inds)):
                ins = np.in1d(face,del_inds)
                outs = np.in1d(face,keep_inds)
                for j in range(3):
                    if ins[j-1] and outs[j]:
                        x0 = coords[face[j]]
                        x1 = coords[face[j-1]]
                        i0,i1 = j,j-1
                    elif ins[j] and outs[j-1]:
                        x0 = coords[face[j-1]]
                        x1 = coords[face[j]]
                        i0,i1 = j-1,j
                    else: 
                        x0,x1,i0,i1 = None,None,-1,-1
                    if x0 is not None:
                        dir_ = x1 - x0 #coords[face[j]] - coords[face[j-1]]
                        len_ = np.linalg.norm(dir_)
                        
                        #Default values
                        shifted[face[i1]] = 1
                        t_hit = arr([0.,0.,0.])
                        if len_>0:
                            dir_ /= len_
                            ray_dirs = np.concatenate([x0,dir_])
                            rays = o3d.core.Tensor([ray_dirs],dtype=o3d.core.Dtype.Float32)
                            ans = scene.cast_rays(rays)
                            intersections = ans['geometry_ids'].numpy().astype('int8')
                            intersections[intersections==scene.INVALID_ID] = -1
                            if intersections[0]!=-1:
                                t_hit = ans['t_hit'].numpy()                        
                                shift_coords[face[i1]] = x0 + t_hit*dir_
                                shifted[face[i1]] = 1
                        
                        new_vert.append( x0 + t_hit*dir_ )
                        new_face.append( [face[i0],count] )
                        #new_face.append( [face[i0],count,face[i2]] )
                        count += 1

        new_vert = arr(new_vert)

        # Keep shifted vertices
        #breakpoint()
        if np.sum(shifted)>0:
            sind = arr([i for i in del_inds if shifted[i]==1])
            if len(sind)>0:
                keep_inds = np.sort(np.concatenate([keep_inds,sind]))
                del_inds = arr([i for i in del_inds if shifted[i]==0])
            coords = shift_coords
                    
        nv = keep_inds.shape[0]
        new_inds = np.linspace(0,nv-1,nv,dtype='int')
        new_faces = arr([f for f in faces if not np.any(np.in1d(f,del_inds))])
        unq = np.unique(new_faces)
        new_faces_refac = new_faces.copy()
        for i,ind in enumerate(keep_inds):
            new_faces_refac[new_faces==ind] = new_inds[i]
            
        #breakpoint()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(coords[keep_inds])
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals[keep_inds])
        mesh.triangles = o3d.utility.Vector3iVector(new_faces_refac)
        
        if False:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[keep_inds])        
            pcd.paint_uniform_color([0,0,1])
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(coords[del_inds])
            #pcd1.points = o3d.utility.Vector3dVector(new_vert)                
            pcd1.paint_uniform_color([1,0,0])
            mesh.paint_uniform_color([1,0,0])
            #o3d.visualization.draw_geometries([target_mesh,mesh,pcd],mesh_show_wireframe=True,mesh_show_back_face=True)
            o3d.visualization.draw_geometries([mesh,pcd,pcd1],mesh_show_wireframe=True,mesh_show_back_face=True)

        #breakpoint()        
        return mesh, keep_inds, new_inds
        
    def simulate_optic_nerve(self,layers=None,layer_thickness=None,ssd=uni(300.,600.),vessel_depth=5000.,fw=None,shoulder_sd=None,flatness=0.5,gap=2.,boundary_height=[0.,0.],nproj=1001,nr=50,rng_nsd=4,plot=False,height=None,depth=None,shoulder_height=arr([50.,50.]),plot_projection=False,add_noise=True):
    
        print('Simulating optic nerve geometry...')
        
        """
        Requires CS and GCL layers as a minimum
        """
        
        if layers is None:
            layers = self.layers
            nlayer = len(layers)
            nsurface = nlayer + 1
            layer_thickness = self.layer_thickness
        
        nlayer = len(layers)
        nsurface = nlayer + 1
        
        # Define geometry
        if shoulder_sd is None:
            shoulder_sd = [ssd+norm(0.,50.),ssd+norm(0.,50.)]
        if fw is None:
            fw = self.eye.optic_cup_diameter # self.eye.optic_disc_radius*uni(0.15,0.3)
        if height is None:
            stp = layers.index('CS')
            height = np.sum(layer_thickness[0:stp])

        if depth is None:
            depth = np.zeros(nsurface) + height + uni(-150,150.)
        depth[layers.index('GCL')] = vessel_depth
        depth[layers.index('vessels')] = vessel_depth
        depth[layers.index('GCL')+2:] = height * 0.2
        
        flatness_v = np.zeros(nsurface) + 0.1
        flatness_v[layers.index('GCL')] = 0.5
        flatness_v[layers.index('vessels')] = 0.5
        
        # Width
        fwc = np.zeros(nsurface)
        fwc[0] = fw
        for i in range(1,nsurface):
            if i in [layers.index('GCL'),layers.index('vessels')]:
               fwc[i] = fwc[i-1]*0.9
            else:
                fwc[i] = fwc[i-1]*1.1
        
        centre = self.eye.optic_disc_centre
        
        fovea_proj = []
        
        last_height, last_width = None, None
        
        sh_scale = np.ones(nsurface)
        for i in range(1,nsurface):
            sh_scale[i] = sh_scale[i-1] * 0.8
        shc = arr([shoulder_height * sh_scale[i] for i in range(nsurface)])
            
        intersection_surface = None
        asymmetric = False # Thicken right-hand shoulder
        
        surface_coords = [[] for _ in range(nsurface)]
        surface_normals = [[] for _ in range(nsurface)]
        surface_faces = [[] for _ in range(nsurface)]
        optic_disc_edge = [[] for _ in range(nsurface)]
        
        displacement = np.concatenate([[0.],np.cumsum(layer_thickness)])
        
        for i in trange(nsurface):
        
            shoulder_sd_c = shoulder_sd.copy()
            # Make right-hand shoulder (i.e. closest to macula) higher and wider
            if i<layers.index('GCL') and asymmetric:
                shoulder_sd_c[1] *= 2 
                shc[i,1] *= np.clip(norm(1.2,0.05),1.01,None) # Commented out to avoid overlap too
            
            coords, faces, edge_verts, normals, mesh = fovea_sim.fovea_sim(nr=nr,nproj=nproj,shoulder_height=shc[i],fw=fwc[i],rng_nsd=rng_nsd,dr=depth[i],flatness=flatness_v[i],boundary_height=boundary_height,shoulder_sd=shoulder_sd_c,plot=False,hole=False,intersection_surface=intersection_surface)

            # Displace & translate
            coords[:,2] -= displacement[i]
            coords[:,0:2] += [centre[0],centre[1]]
            # Rotate 90 deg around z (for asymmetric meshes, if required)
            #angle = np.deg2rad(90.)
            #R = np.eye(3)
            #R[0,0:2] = [np.cos(angle), -np.sin(angle)]
            #R[1,0:2] = [np.sin(angle),  np.cos(angle)]
            #o = np.atleast_2d(arr([0.,0.,0.]))
            #coords = np.squeeze((R @ (coords.T-o.T) + o.T).T)
            coords_copy = coords.copy()
            coords_copy[:,0] = coords[:,1]
            coords_copy[:,1] = coords[:,0]
            coords = coords_copy
            
            if i==layers.index('GCL'):
                intersection_surface = coords.copy()
            elif i>layers.index('GCL') and i!=layers.index('vessels'):
                #breakpoint()
                src_mesh = o3d.geometry.TriangleMesh()
                src_mesh.vertices = o3d.utility.Vector3dVector(coords)
                src_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
                src_mesh.triangles = o3d.utility.Vector3iVector(faces)
                
                target_ind = layers.index('GCL')
                target_mesh = o3d.geometry.TriangleMesh()
                target_mesh.vertices = o3d.utility.Vector3dVector(surface_coords[target_ind])
                target_mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[target_ind])
                target_mesh.triangles = o3d.utility.Vector3iVector(surface_faces[target_ind])

                mesh, keep_inds, new_inds = self.delete_faces_below_target_mesh(src_mesh, target_mesh, edge_verts=edge_verts)
                coords = np.asarray(mesh.vertices)
                normals = np.asarray(mesh.vertex_normals)
                faces = np.asarray(mesh.triangles)
                
                edge_verts = np.asarray(edge_verts)
                #if not np.all(np.in1d(edge_verts,keep_inds)):
                #    breakpoint()
                edge_verts_refac = edge_verts.copy()
                #breakpoint()
                for j,ind in enumerate(keep_inds):
                    edge_verts_refac[edge_verts==ind] = new_inds[j] 
                edge_verts = edge_verts_refac
            
            optic_disc_edge[i] = coords[edge_verts]

            surface_coords[i] = coords
            surface_normals[i] = normals
            surface_faces[i] = faces
        
        if False: # plot:      
            mn,mx = coords.min(axis=0),coords.max(axis=0)
            cols = np.zeros([nsurface,3],dtype='int16')
            cols[:,0] = 1
            gcl_ind = layers.index('GCL')
            cols[gcl_ind,:] = [0,0,1]
            v_ind = layers.index('vessels')
            cols[v_ind,:] = [0,1,0]        
            z0 = -1e12
            size = nproj
            meshes = self.projection(arr([mn[0],0.,z0]),arr([mx[0],0.,z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size)
            breakpoint()
        
            meshes = []
            rmp = np.linspace(0.,1.,nsurface)
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                #mesh.paint_uniform_color([rmp[i], 0, np.flip(rmp[i])])
                meshes.append(mesh)
                
            o3d.visualization.draw_geometries(meshes,mesh_show_back_face=True,mesh_show_wireframe=True)
            
            
        if plot:
            meshes = []
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                meshes.append(mesh)
            
            o3d.visualization.draw_geometries(meshes,mesh_show_back_face=True,mesh_show_wireframe=True)
            #breakpoint()   
        
        if plot_projection:
            mn,mx = coords.min(axis=0),coords.max(axis=0)
            cols = np.zeros([nsurface,3],dtype='int16')
            cols[:,0] = 1
            gcl_ind = layers.index('GCL')
            cols[gcl_ind,:] = [0,0,1]
            v_ind = layers.index('GCL')
            cols[v_ind,:] = [0,1,0]        
            z0 = -1e6
            size = 1024
            
            fig, axs = plt.subplots(2,figsize=[12,12])
            no_line = True
            # X-axis
            meshes = self.projection(arr([mn[0],self.eye.optic_disc_centre[1],z0]),arr([mx[0],self.eye.optic_disc_centre[1],z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size,grey_val=self.grey_val,no_line=no_line,axis=axs[0],show=False)
            meshes = self.projection(arr([self.eye.optic_disc_centre[0],mn[1],z0]),arr([self.eye.optic_disc_centre[0],mx[1],z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size,grey_val=self.grey_val,no_line=no_line,axis=axs[1],show=False)
            
            axs[0].set(xlabel='x')
            axs[0].set_xlim([self.eye.optic_disc_centre[0]-1000,self.eye.optic_disc_centre[0]+1000])
            axs[1].set(xlabel='y')
            axs[1].set_xlim([self.eye.optic_disc_centre[1]-1000,self.eye.optic_disc_centre[1]+1000])
            
            plt.show()            
 
        return surface_coords, surface_normals, surface_faces, optic_disc_edge
        
    def projection(self, A, B, meshes=None, surface_coords=None, surface_normals=None, surface_faces=None, size=256, colors=None, xrange=None, yrange=None, plot_file=None, grey_val=None, no_line=False,axis=None,show=True):
    
        """
        Create 2D projection within a plane defined by A and B vertices
        """
    
        import open3d as o3d

        # Define meshes if surface coords have been passed instead
        # Should be a list of nx3 numpy arrays
        if meshes is None:
            if surface_coords is None:
                return None
            meshes = []
            nsurface = len(surface_coords)
            rmp = np.linspace(0.,1.,nsurface)
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                #mesh.paint_uniform_color([rmp[i], 0, np.flip(rmp[i])])
                meshes.append(mesh)
         
        if not np.any(A==B):
            print('Not a plane!')
            return None
        if A[2]==B[2]: # z-projection
            x = np.linspace(A[0],B[0],size)
            y = np.linspace(A[1],B[1],size)
            z = np.zeros(size) + A[2] #arr([A[2],B[2]])
        elif A[1]==B[1]:
            x = np.linspace(A[0],B[0],size)
            y = arr([A[1],B[1]])
            z = np.linspace(A[2],B[2],size)
        elif A[0]==B[0]:
            x = arr([A[0],B[0]])
            y = np.linspace(A[1],B[1],size)
            z = np.linspace(A[2],B[2],size)
            
        #breakpoint()

        if plot_file is not None:
            figsize = (16,16)
        else:
            figsize = (8,8)
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = plt.gca()
        
        if grey_val is not None:
            axis.set_facecolor('black')
        
        x0_store = []
        
        for i,mesh in enumerate(meshes):
            scene = o3d.t.geometry.RaycastingScene()  
            mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

            dir_ = np.tile([0,0,1],(size,1))
            ray_dirs = np.vstack([x,y,z,dir_[:,0],dir_[:,1],dir_[:,2]]).transpose()
            # Convert to tensors
            rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays)
            x0 = ans['t_hit'].numpy() + A[2]
            
            if colors is not None:
                col = colors[i]
            else:
                col = None
                
            if not no_line:
                if A[1]==B[1]:
                    plt.plot(x,x0,color=col)
                else:
                    plt.plot(y,x0,color=col)
            x0_store.append(x0)
            
            if i>0 and grey_val is not None:
                col = str(grey_val[i-1])
                if A[1]==B[1]:
                    axis.fill_between(x, x0_store[i-1], x0_store[i],color=col)
                else:
                    axis.fill_between(y, x0_store[i-1], x0_store[i],color=col)

        if xrange is not None:            
            plt.xlim(xrange)
        if yrange is not None:            
            plt.ylim(yrange)
            
        if axis is None:
            axis = plt.gca()
        axis.set_aspect('equal')
            
        if show is True or plot_file is not None:
            if plot_file is None:
                plt.show()
            else:
                plt.savefig(plot_file)
        
        return meshes

    def create_grid_fill(self,x,y,xv,yv,layers,layer_thickness,n=50,fill_mode='rectangular',domain_mode='circular',add_central_line=False,plot=False):
    
        print('Creating grid fill')
        
        nlayer = len(layers)
        nsurface = nlayer + 1
        
        surface_coords = [[] for _ in range(nsurface)]
        surface_normals = [[] for _ in range(nsurface)]
        surface_faces = [[] for _ in range(nsurface)]
        
        for i in trange(nsurface):
            # Displace
            if i==0:
                thk = 0
            else:
                thk = np.sum(layer_thickness[:i])
            
            # Create grid points
            if fill_mode=='rectangular':
                zg = np.zeros([n,n]) - thk
                xg = np.linspace(x.min(),x.max(),n)
                yg = np.linspace(y.min(),y.max(),n)
                xvg, yvg = np.meshgrid(xg, yg, sparse=False, indexing='ij')
                points = arr([xv.flatten(),yv.flatten(),zg.flatten()]).transpose()
                normals = np.zeros([xv.size,3])
                normals[:,2] = 1.
                
                # Define faces
                tri = Delaunay(np.vstack([xvg.flatten(),yvg.flatten()]).transpose()) # take the first two dimensions
                faces = tri.simplices
                
                # Make outer boundary circular
                if domain_mode=='circular':
                    centre = self.eye.occular_centre[0:2]
                    rad = x.max()-centre[0]
                    outside = arr([np.linalg.norm(p[0:2]-centre)>rad for p in points])
                    # Remove faces connecting removed points
                    del_inds = np.where(outside)
                    if len(del_inds)>0:
                        faces,retina_edge_inds = self.delete_faces_containing_vertices(faces,del_inds[0],points.shape[0])
                    points = points[~outside]
                    normals = normals[~outside]
                    
                    # Create neat edge at exact radius
                    tg = np.linspace(-np.pi,np.pi,n)
                    #rg = np.zeros(n) + rad
                    
                    xv_edge = arr([rad*np.cos(t) for t in tg])
                    yv_edge = arr([rad*np.sin(t) for t in tg])
                    zv_edge = np.zeros(xv_edge.shape[0]) - thk

                    points_edge = np.vstack([xv_edge,yv_edge,zv_edge]).transpose()
                    normals_edge = np.zeros([xv_edge.size,3])
                    normals_edge[:,2] = 1.
                    ordered_grid_edge, grid_srt = self.sort_points_clockwise(points_edge,return_sort=True)

                    # Stitch edge to main mesh
                    feature_edge = points[retina_edge_inds]
                    ordered_feature_edge,feature_srt = self.sort_points_clockwise(feature_edge,return_sort=True)
                    stitch_faces = self.stitch_edges(ordered_feature_edge,ordered_grid_edge,plot=False,sort=False)
                    
                    # Create vertex index lookup table to reference vertices when concatonated below (feature,grid)
                    combined_edge_ind_lookup = np.concatenate([retina_edge_inds[feature_srt],np.linspace(0,points_edge.shape[0]-1,points_edge.shape[0],dtype='int')+points.shape[0]])
                    
                    # Add in new boundary points
                    points = np.concatenate([points,ordered_grid_edge])
                    normals = np.concatenate([normals,normals_edge])
                    
                    # Refactor vertex indices to point cloud
                    stitch_faces_refact = stitch_faces.copy()
                    for f in range(combined_edge_ind_lookup.shape[0]):
                        stitch_faces_refact[stitch_faces==f] = combined_edge_ind_lookup[f] #+ points.shape[0]

                    if False:
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(points) 
                        mesh.triangles = o3d.utility.Vector3iVector(stitch_faces_refact)
                        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True,mesh_show_wireframe=True)
                    
                    faces = np.concatenate([faces,stitch_faces_refact])
                    
            elif fill_mode=='circular':

                tg = np.linspace(-np.pi,np.pi,n)
                #rg = np.linspace(x.min(),x.max(),n)
                rg = np.linspace(0,x.max(),n)
                xv = arr([r*np.cos(t) for r in rg for t in tg])
                yv = arr([r*np.sin(t) for r in rg for t in tg])
                zv = np.zeros(xv.shape[0]) - thk

                points = arr([xv,yv,zv]).transpose()
                normals = np.zeros([xv.size,3])
                normals[:,2] = 1.
                
                # Define faces (from fovea_sim.py)
                faces = []
                nproj = rg.shape[0]
                nangle = tg.shape[0]
                for j in range(1,tg.shape[0]):
                    for ii in range(rg.shape[0]):
                        a,b = ii+((j-1)*nproj), ii+((j-1)*nproj)+1,
                        c,d = ii+(j*nproj)+1,   ii+(j*nproj)
                        faces.append(self.ensure_face_direction(arr([a,c,b]), points[[a,b,b]] ))
                        faces.append(self.ensure_face_direction(arr([a,c,d]), points[[a,c,d]] ))
                faces = arr(faces)
                
            # Add central line for optic nerve plot
            if add_central_line:
                ax1 = arr([np.zeros(ny)+self.eye.fovea_centre[0],y,np.zeros(ny) - thk]).transpose()
                ax2 = arr([x,np.zeros(nx)+self.eye.fovea_centre[1],np.zeros(ny) - thk]).transpose()
                points = np.concatenate([points,ax1,ax2])
                normals_clx = np.zeros([ax1.shape[0],3])
                normals_clx[:,2] = 1.
                normals_cly = np.zeros([ax2.shape[0],3])
                normals_cly[:,2] = 1.
                normals = np.concatenate([normals,normals_clx,normals_cly])
                
            surface_coords[i] = points
            surface_normals[i] = normals
            surface_faces[i] = faces
            
        if plot:
            meshes = []
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                meshes.append(mesh)
            
            o3d.visualization.draw_geometries(meshes,mesh_show_back_face=True,mesh_show_wireframe=True)
                
        print('Grid fill complete')
        return surface_coords, surface_normals, surface_faces              

    def sort_points_clockwise(self, points, return_sort=False):
    
        com = np.mean(points[:,0:2],axis=0)
        ang = np.arctan2(points[:,0]-com[0],points[:,1]-com[1])
        srt = np.argsort(ang)
        
        if return_sort:
            return points[srt], srt
        else:
            return points[srt]

    def embed_feature_in_grid(self, layers, grid_points, grid_normals, grid_faces, feature_edge, feature_coords, feature_normals, feature_faces, plot=False, stitch=True,convex_hull=True,remove_bounding_box_faces=True):   
    
        nlayer = len(layers)
        nsurface = nlayer + 1
        
        surface_coords = [[] for _ in range(nsurface)]
        surface_normals = [[] for _ in range(nsurface)]
        surface_faces = [[] for _ in range(nsurface)]
        
        for i in trange(nsurface):
        
            points = grid_points[i]      
            normals = grid_normals[i]  

            # Order feature edge points
            if convex_hull:
                hull = ConvexHull(feature_edge[i][:,0:2])
                ordered_feature_edge = feature_edge[i][hull.vertices]
            else:
                ordered_feature_edge = self.sort_points_clockwise(feature_edge[i])
            # Find surface points that fall inside the feature
            poly = Polygon(ordered_feature_edge[:,0:2])
            inside = arr([Point(x[0],x[1]).within(poly) for x in points[:,0:2]])
            
            # Remove faces connecting removed points
            # !!! Assumes grid is finely spaced compared with size of feature !!!
            del_inds = np.where(inside)
            if len(del_inds)>0:
                # Delete grid faces and also store the vertex indices at the edge of the hole
                # N.B. grid_faces is no longer a valid set of faces as it references deleted vertices
                del_inds = del_inds[0]
                #print(f'Deleting {del_inds.shape[0]} faces...')
                grid_faces_outside_feature, grid_edge_inds = self.delete_faces_containing_vertices(grid_faces[i],del_inds,points.shape[0])
                
                # Delete points and normals that are inside the feature
                points = points[~inside]
                normals = normals[~inside]
                               
                # Remove faces from additional grid points to make regridding easier
                # SLOW....!
                if remove_bounding_box_faces:
                    #print('Removing bounding box faces...')
                    bbox = arr([ [ np.min(ordered_feature_edge[:,0]),np.max(ordered_feature_edge[:,0]) ],
                                 [ np.min(ordered_feature_edge[:,1]),np.max(ordered_feature_edge[:,1]) ],
                                 [ np.min(ordered_feature_edge[:,2]),np.max(ordered_feature_edge[:,2]) ] ])

                    # Isolate points within a bounding box                                 
                    dx,dy = 250.,250. # um
                    bbox_inds = np.where( (np.all(points[:,0:2]>=bbox[0:2,0]-arr([dx,dy]),axis=1)) & (np.all(points[:,0:2]<=bbox[0:2,1]+arr([dx,dy]),axis=1)) )[0]
                    bbox_points = points[bbox_inds]

                    #breakpoint()
                    # TEST
                    sind = (np.in1d(grid_faces_outside_feature[:,0],bbox_inds)) & (np.in1d(grid_faces_outside_feature[:,1],bbox_inds)) & (np.in1d(grid_faces_outside_feature[:,2],bbox_inds)) 
                    # ORIG
                    #sind = arr([np.all([f[0] in bbox_inds,f[1] in bbox_inds,f[2] in bbox_inds]) for j,f in enumerate(grid_faces_outside_feature)])
                    grid_faces_outside_feature = grid_faces_outside_feature[~sind]
                    grid_edge_inds = bbox_inds

                # Combine vertices
                surface_coords[i] = np.concatenate([points,feature_coords[i]])
                surface_normals[i] = np.concatenate([normals,feature_normals[i]])
                surface_faces[i] = np.concatenate([grid_faces_outside_feature,feature_faces[i]+points.shape[0]])
                
                # Stitch edges together
                if stitch:
                    #print('Stitching edges')
                    # Get coordinates for points at the internal edge of the grid
                    grid_edge = points[grid_edge_inds]
                    # Order those points
                    ordered_grid_edge = self.sort_points_clockwise(grid_edge)
                    # Stitch
                    stitch_faces = self.stitch_edges(ordered_feature_edge,ordered_grid_edge,plot=False,sort=False)

                    ne1,ne2 = ordered_feature_edge.shape[0],ordered_grid_edge.shape[0]
                    
                    # Locate feature points in grid point cloud
                    combined_edge = np.concatenate([ordered_feature_edge,ordered_grid_edge])
                    # BOTTLENECK! Would be better to track indices of edge vertices, but for now just locate them by brute force
                    combined_edge_ind_lookup = arr([np.where(np.all(surface_coords[i]==e,axis=1))[0][0] for e in combined_edge]).squeeze()
                    
                    # Refactor vertex indices to point cloud
                    stitch_faces_refact = stitch_faces.copy()
                    for f in range(combined_edge.shape[0]):
                        stitch_faces_refact[stitch_faces==f] = combined_edge_ind_lookup[f] #+ points.shape[0]

                    if False: # plot
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                        mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                        mesh.triangles = o3d.utility.Vector3iVector(grid_faces_outside_feature)
                        
                        mesh1 = o3d.geometry.TriangleMesh()
                        mesh1.vertices = o3d.utility.Vector3dVector(feature_coords[i])
                        mesh1.vertex_normals = o3d.utility.Vector3dVector(feature_normals[i])
                        mesh1.triangles = o3d.utility.Vector3iVector(feature_faces[i])
                        
                        mesh2 = o3d.geometry.TriangleMesh()
                        mesh2.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                        #mesh2.vertex_normals = o3d.utility.Vector3dVector(feature_normals[i])
                        mesh2.triangles = o3d.utility.Vector3iVector(stitch_faces_refact)
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        
                        pcd2 = o3d.geometry.PointCloud()
                        pcd2.points = o3d.utility.Vector3dVector(bbox_points)
                        
                        pcd1 = o3d.geometry.PointCloud()
                        pcd1.points = o3d.utility.Vector3dVector(ordered_feature_edge)
                        
                        o3d.visualization.draw_geometries([mesh,mesh1,mesh2,pcd1,pcd2],mesh_show_back_face=True,mesh_show_wireframe=True)
                        breakpoint()

                    # Concatenate with previous faces
                    surface_faces[i] = np.concatenate([grid_faces_outside_feature,feature_faces[i]+points.shape[0],stitch_faces_refact])
            else:
                breakpoint()   
                
        if plot:
            meshes = []
            for i in range(nsurface):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
                mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
                mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
                meshes.append(mesh)
            
            o3d.visualization.draw_geometries([meshes[-1]],mesh_show_back_face=True,mesh_show_wireframe=True) 
            
        return surface_coords, surface_normals, surface_faces  

    def create_retina_surface(self, plot=False,nr=250,nproj=1001,n=500,rng_nsd=4,simulate_macula=True,simulate_optic_nerve=True,add_grid_fill=True,plot_file=None,add_simplex_noise=True,project=True,filename=None,domain_mode='circular',fill_mode='rectangular',regrid=True,ofile=None,domain=None,vessel_depth=5000.):
        
        if domain is not None:
            self.set_domain(domain)
        
        # Grid params
        nx,ny = n,n
        dx = (self.domain[0,1]-self.domain[0,0])/float(nx)
        dy = (self.domain[1,1]-self.domain[1,0])/float(ny)
        x = np.linspace(self.domain[0,0],self.domain[0,1],nx)
        y = np.linspace(self.domain[1,0],self.domain[1,1],ny)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
        
        surfaces = np.zeros([self.nsurface,n,n])
        surface_coords = [[] for _ in range(self.nsurface)]
        surface_normals = [[] for _ in range(self.nsurface)]
        surface_faces = [[] for _ in range(self.nsurface)]
        
        # Macula
        macula_edge = []
        if simulate_macula:
            # Reduce the number of projection points and angles for the macula 
            nproj_mac = np.max([int(nproj / 10),10])
            nr_mac = np.max([int(nr / 5),10])
            m_surface_coords, m_surface_normals, m_surface_faces, macula_edge = self.simulate_macula(nproj=nproj_mac,nr=nr_mac,rng_nsd=rng_nsd)

        # Optic nerve
        optic_disc_edge = []
        if simulate_optic_nerve:
            od_surface_coords, od_surface_normals, od_surface_faces, optic_disc_edge = self.simulate_optic_nerve(self.layers,self.layer_thickness,nproj=nproj,nr=nr,rng_nsd=rng_nsd,shoulder_height=arr([500.,500.]),vessel_depth=vessel_depth)
                
        # Create retina grid (flat)
        # TODO: Blend optic disc into macula
        if add_grid_fill:
            g_surface_coords, g_surface_normals, g_surface_faces = self.create_grid_fill(x,y,xv,yv,self.layers,self.layer_thickness,n=n,domain_mode=domain_mode)
            
            if simulate_macula:
                print('Stitching macula model into grid...')
                g_surface_coords, g_surface_normals, g_surface_faces = self.embed_feature_in_grid(self.layers, g_surface_coords, g_surface_normals, g_surface_faces, macula_edge, m_surface_coords, m_surface_normals, m_surface_faces, plot=False)
            
            if simulate_optic_nerve:
                print('Stitching optic nerve model into grid...')
                g_surface_coords, g_surface_normals, g_surface_faces = self.embed_feature_in_grid(self.layers, g_surface_coords, g_surface_normals, g_surface_faces, optic_disc_edge, od_surface_coords, od_surface_normals, od_surface_faces)
             
            surface_coords = g_surface_coords
            surface_normals = g_surface_normals
            surface_faces = g_surface_faces             
        else:
            surface_coords = m_surface_coords
            surface_normals = m_surface_normals
            surface_faces = m_surface_faces        
                   
        if False: # Plot projection (prior to projection)
            #mn,mx = surface_coords[0].min(axis=0),surface_coords[0].max(axis=0)
            mn,mx = self.eye.optic_disc_centre-arr([3000.,3000.,0.]), self.eye.optic_disc_centre+arr([7000.,7000.,0.])
            cols = np.zeros([self.nsurface,3],dtype='int16')
            cols[:,0] = 1
            gcl_ind = self.layers.index('GCL')
            cols[gcl_ind,:] = [0,0,1]
            v_ind = self.layers.index('vessels')
            cols[v_ind,:] = [0,1,0]        
            z0 = -1e6
            size = 1024
            meshes = self.projection(arr([mn[0],0.,z0]),arr([mx[0],0.,z0]),surface_coords=surface_coords,surface_normals=surface_normals,surface_faces=surface_faces,colors=cols,size=size)
                    
        # Simplex noise
        if add_simplex_noise:
            print('Adding simplex noise')
            
            layer_noise_feature_size = [norm(200.,25.),norm(1500.,50.)] # um
            layer_noise_scale_factor = 0.025 # scales layer thickness (reduced from 0.05)
            
            common_noise_feature_size = [norm(100,10.),norm(2000.,100.)] # um 
            common_noise_scale_factor = [uni(1.,10.),uni(15.,50.)] # scales normalised noise
            #common_noise_scale_factor = [uni(1.,20.),uni(25.,100.)] # scales normalised noise # Large!
            #common_noise_scale_factor = [uni(2.,40.),uni(50.,200.)] # scales normalised noise # Large!
            #common_noise_scale_factor = [uni(2.,20.),uni(2000.,100.)] # scales normalised noise # Very large!
            
            # Noise on each layer, independent
            print('Adding independent layer noise')
            for i,coords in enumerate(tqdm(surface_coords)):
                seedx = uuid.uuid1().int>>64
                simplex_x = OpenSimplex(seedx)
                noise = np.zeros(coords.shape[0])
                for ii,crd in enumerate(coords):
                    for k,sc in enumerate(layer_noise_feature_size):
                        noise[ii] += simplex_x.noise2(crd[0]/sc,crd[1]/sc)
            
                noise = ((noise-noise.min()) * 2 / (noise.max()-noise.min())) - 1.
                if i<len(surface_coords)-1:
                    noise_scale = self.layer_thickness[i]*layer_noise_scale_factor #100. # um
                else:
                    noise_scale = self.layer_thickness[i-1]*layer_noise_scale_factor
                coords[:,2] += noise * noise_scale
 
            # Noise common to all layers
            seedx = uuid.uuid1().int>>64
            print('Adding common layer noise')
            for i,coords in enumerate(tqdm(surface_coords)):
                simplex_x = OpenSimplex(seedx)
                noise = np.zeros(coords.shape[0])
                for ii,crd in enumerate(coords):
                    for k,sc in enumerate(common_noise_feature_size):
                        noise[ii] += simplex_x.noise2(crd[0]/sc,crd[1]/sc) * common_noise_scale_factor[k]
            
                # Rescale between -1 and +1
                #noise = ((noise-noise.min()) * 2 / (noise.max()-noise.min())) - 1.
                coords[:,2] += noise #* common_noise_scale_factor
                    
        # Project onto sphere
        if project: #True:
            #breakpoint()
            print('Projecting onto sphere')
            domain_size = self.domain[:,1] - self.domain[:,0]
            count = 0
            for i,coords in enumerate(tqdm(surface_coords)):  
                old_coords = coords.copy()
                #print(count)
                count += 1
                
                cur_norms = []
                
                for j,crd in enumerate(coords):                        
                    #print(i,len(coords))
                    # Draw vector from centre of the eye to coords
                    surface_height = 0.
                    P = self.eye.occular_centre - arr([crd[0],crd[1],surface_height])
                    # Rotation at retina surface
                    rt = geometry.align_vectors(arr([0.,0.,1.]),P)
                    # Translate point to surface, rotate, then translate back
                    surface_coords[i][j] = np.dot(rt,crd)
                    surface_normals[i][j] = np.dot(rt,surface_normals[i][j])

        # Write layers as STL (or PLY) files
        ofiles = [ofile.replace('.ply',f'_{self.layers[i]}.ply') for i,x in enumerate(self.layers)]
        dir_ = os.path.dirname(ofile)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        meshes = []
        for i in range(len(self.layers)):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(surface_coords[i])
            mesh.vertex_normals = o3d.utility.Vector3dVector(surface_normals[i])
            mesh.triangles = o3d.utility.Vector3iVector(surface_faces[i])
            mesh.compute_triangle_normals()
            meshes.append(mesh)
            
            self.write_surface_mesh(mesh,ofiles[i])
        
        #breakpoint()
        if plot: # Plot projection       
            #mn,mx = surface_coords[0].min(axis=0),surface_coords[0].max(axis=0)
            mn,mx = self.eye.optic_disc_centre-arr([3000.,3000.,0.]), self.eye.optic_disc_centre+arr([7000.,7000.,0.])
            cols = np.zeros([self.nsurface,3],dtype='int16')
            cols[:,0] = 1
            gcl_ind = self.layers.index('GCL')
            cols[gcl_ind,:] = [0,0,1]
            v_ind = self.layers.index('GCL')
            cols[v_ind,:] = [0,1,0]        
            z0 = -1e6
            size = 1024
            meshes = self.projection(arr([mn[0],0.,z0]),arr([mx[0],0.,z0]),meshes=meshes,colors=cols,size=size,plot_file=plot_file,grey_val=self.grey_val,no_line=True)
            
        # Create surfaces (assuming non-degenerate, for now)
        if regrid:
            print('Regridding surfaces')
            for i,coords in enumerate(surface_coords):
                #print(i)
                surfaces[i] = griddata(coords[:,0:2], coords[:,2], (xv,yv), method='nearest') #.transpose()
                    
            # Create vessel surface (primarily in the ganglion cell layer (layer below the NFL) https://www.nature.com/articles/srep42201#:~:text=1)3.,ganglion%20cell%20layer%20(GCL).
            print('Creating vessel surface')
            #vessel_surf = surfaces[layers.index('vessels')] # + surfaces[layers.index('GCL')+1]) / 2.
        
        sc = 255./2.
        self.vals = (arr(self.grey_val) * sc).astype('int')
        #self.grey_val = arr(grey_val)
        #self.seg_val = arr(seg_val)
        
        self.surface_coords = surface_coords
        self.surface_nomals = surface_normals
        self.surface_faces = surface_faces
        #self.layers = layers
        #self.layer_thickness = layer_thickness
        
        #if filename is not None:
        #    self.write(filename,xv,yv,vessel_surf,vessel_surf_normals,self.surfaces,vals,arr(seg_val))

        vessel_surf = surface_coords[self.layers.index('vessels')]
        vessel_surf_normals = surface_normals[self.layers.index('vessels')]

        return vessel_surf, vessel_surf_normals, xv, yv, self.vals, self.seg_val, ofiles
        
    def write(self,filename,xv,yv,vessel_surf,normals,surfaces,vals,seg_val):
        np.savez(filename,xv,yv,vessel_surf,normals,surfaces,vals,seg_val)
        
    def write_surface_mesh(self, mesh, ofile=''):
        o3d.io.write_triangle_mesh(ofile,mesh)
        
    def load_surface(self,filename):
        import open3d as o3d
        return o3d.io.read_triangle_mesh(filename)
        
    def load_surfaces(self,path):
        meshes = []
        for lname in self.layers:
            cpath = join(path,f'retina_surface_{lname}.ply')
            meshes.append(self.load_surface(cpath))
        return meshes
        
