import numpy as np
arr = np.asarray
norm = np.linalg.norm
import sys
import os
join = os.path.join
from skimage import draw
import nibabel as nib
from vessel_sim import geometry
from matplotlib import pyplot as plt
from functools import partial
import open3d as o3d
from pymira import spatialgraph

# GRID EMBEDDING

class EmbedVessels(object):

    def __init__(self,ms=256,vessel_grid_size=None,domain=None,resolution=2.5,offset=arr([0.,0.,0.]),store_midline=True,store_midline_diameter=False,store_diameter=False):
    
        """
        Class for embedding spatial graphs into 3D arrays
        ms = 3D array size
        resolution = spatial resolution
        offset = centre of domain (um)
        """
        
        self.ms = ms # n pixels
        
        self.dtype = 'int16'
        
        if vessel_grid_size is None:
            if type(ms)==int:
                self.vessel_grid_size = arr([ms,ms,ms])
            else:
                self.vessel_grid_size = arr(ms)
        else:
            self.vessel_grid_size = np.asarray(vessel_grid_size)
            
        # Get domain size and resolution from domain and grid size (and ignore any passed resolution).
        # Otherwise use the resolution as passed
        if domain is not None:
            self.domain = domain
            self.domain_size = (self.domain[:,1] - self.domain[:,0]).squeeze()
            self.resolution = self.domain_size / self.vessel_grid_size
            self.domain_offset = None
        else:
            self.domain_offset = offset
            self.resolution = arr([resolution,resolution,resolution]) # um
            self.domain_size = self.vessel_grid_size * self.resolution # um
        
        if self.domain_offset is not None:
            self.domain = arr([[(-self.domain_size[0]/2) + self.domain_offset[0], (self.domain_size[0]/2) + self.domain_offset[0]],
                               [(-self.domain_size[1]/2) + self.domain_offset[1], (self.domain_size[1]/2) + self.domain_offset[1]],
                               [self.domain_offset[2]-(self.domain_size[2]/2), self.domain_offset[2]+(self.domain_size[2]/2.)]]) # um
        elif self.domain is None:
            self.domain = arr( [ [0,self.domain_size[0]], [0,self.domain_size[1]], [0,self.domain_size[2]] ] )
        #self.domain[:,0] += self.domain_offset
        #self.domain[:,1] += self.domain_offset
        self.vessel_grid = np.zeros(self.vessel_grid_size,dtype='float')
        #self.o2_grid = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        
        if store_midline:
            self.vessel_midline = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        else:
            self.vessel_midline = None
        if store_diameter:
            self.vessel_diameter = np.zeros(self.vessel_grid_size,dtype='float')
        else:
            self.vessel_diameter = None
        if store_midline_diameter:
            self.vessel_midline_diameter = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        else:
            self.vessel_midline_diameter = None
        self.surface_grid = None
        self.label_grid = None
        
        #print('Vessel embedding domain: {}'.format(self.domain_size))

    def coord_to_pix(self,coord,extent_um,dims,clip=True,voxel_size=None,y_flip=True):
        """ Convert 3D spatial coordinates into 3D pixel indices """
        if voxel_size is None:
            dims,extent_um = arr(dims),arr(extent_um)
            voxel_size = (extent_um[:,1]-extent_um[:,0])/dims
        pix = (coord-extent_um[:,0])/voxel_size
        pix = pix.astype('int')
        if clip:
            pix = np.clip(pix,0,dims-1)
        
        if y_flip:
            pix[1] = dims[1] - pix[1] - 1

        return pix
            
    def embed_sphere_aniso(self, x0, r, dims=None,fill_val=None,data=None,resolution=[1,1,1],fill_mode='binary'):
    
        if data is not None:
            dims = data.shape
        else:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,self.dtype)
            
        # Create ellipse
        ellip_base = draw.ellipsoid(r/2.,r/2.,r/2.,spacing=resolution,levelset=False)
        subs = np.where(ellip_base)
        sx = subs[0] + x0[0] - int(np.floor(ellip_base.shape[0]/2))
        sy = subs[1] + x0[1] - int(np.floor(ellip_base.shape[1]/2))
        sz = subs[2] + x0[2] - int(np.floor(ellip_base.shape[2]/2))
        
        sind = np.where((sx>=0) & (sx<dims[0]) & (sy>=0) & (sy<dims[1]) & (sz>=0) & (sz<dims[2]))
        if len(sind[0])==0:
            return data
        sx, sy, sz = sx[sind[0]], sy[sind[0]], sz[sind[0]]
        
        if fill_val is None:
            fill_val = 255
            
        if fill_mode=='binary':
            data[sx,sy,sz] = fill_val
        elif fill_mode=='radius':
            vals = np.zeros(sx.shape[0]) + r
            old_vals = data[sx,sy,sz]
            vals = np.max(np.vstack([vals,old_vals]),axis=0)
            data[sx,sy,sz] = vals
        elif fill_mode=='diameter':
            vals = np.zeros(sx.shape[0]) + r*2
            old_vals = data[sx,sy,sz]
            vals = np.max(np.vstack([vals,old_vals]),axis=0)
            data[sx,sy,sz] = vals
        elif fill_mode=='partial_volume':
            if r>1:
                data[sx,sy,sz] = r
            else:
                data[sx,sy,sz] = r

        return data                    
            
    def embed_sphere(self,x0,r,dims=None,fill_val=None,data=None,fill_mode='binary'):
    
        """
        Embed a sphere in a 3D volume
        x0: centre coordinate (pixels)
        r: radius (pixels)
        """
    
        if data is not None:
            dims = data.shape
        else:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,self.dtype)
            
        r_upr = np.ceil(r).astype('int')
        r_lwr = np.floor(r).astype('int')
            
        z0 = np.clip(x0[2]-r_upr,0,dims[2])
        z1 = np.clip(x0[2]+r_upr,0,dims[2])
        
        if fill_val is None:
            fill_val = 1.

        for i,zp in enumerate(range(z0,z1)):
            dz = np.abs(x0[2]-zp)

            rz = np.square(r_lwr)-np.square(dz)
            if rz>0.:
                xp, yp = draw.disk(x0[0:2],np.sqrt(rz), shape=dims[0:2])
                
                if fill_mode=='binary':
                    data[xp,yp,zp] = 1. #fill_val
                elif fill_mode=='radius':
                    vals = np.zeros(xp.shape[0]) + r
                    old_vals = data[xp,yp,zp]
                    vals = np.max(np.vstack([vals,old_vals]),axis=0)
                    data[xp,yp,zp] = vals
                elif fill_mode=='diameter':
                    vals = np.zeros(xp.shape[0]) + r*2
                    old_vals = data[xp,yp,zp]
                    vals = np.max(np.vstack([vals,old_vals]),axis=0)
                    data[xp,yp,zp] = vals
                elif fill_mode=='partial_volume':
                    if r>1:
                        data[xp,yp,zp] = 1
                    else:
                        data[xp,yp,zp] += r
                    

        return data                     
            
    def embed_cylinder(self,x0i,x1i,r0i,r1i,dims=None,fill_val=None,data=None):
    
        x0 = np.array(x0i,dtype='float')
        x1 = np.array(x1i,dtype='float')
        r0 = np.array(r0i,dtype='float')
        r0_upr = np.ceil(r0i).astype('float')
        r0_lwr = np.floor(r0i).astype('float')
        r1 = np.array(r1i,dtype='float')
        r1_upr = np.ceil(r1i).astype('float')
        r1_lwr = np.floor(r1i).astype('float')
        
        if fill_val is None:
            fill_val = 1. #np.mean([r0,r1]) # Fill value is mean radius

        C = x1 - x0
        length = norm(C)

        X, Y, Z = np.eye(3)
        
        if data is None:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,dtype=self.dtype)
        else:
            dims = data.shape
        nx, ny, nz = data.shape
        
        # If x0=x1, return empty data
        if np.all(C==0.):
            return data

        orient = np.argmin([geometry.vector_angle(np.asarray(x,'float'), C) for x in np.eye(3)])
        if orient==0:
            ao = arr([1,2,0])
            shape = (ny, nz)
        elif orient==1:
            ao = arr([0,2,1])
            shape = (nx, nz)
        else:
            ao = arr([0,1,2])
            shape = (nx, ny)

        theta = geometry.vector_angle(Z, C)
        alpha = geometry.vector_angle(X, x0 + C)
        
        z0_range = x0[ao[2]]-r0*np.sin(theta),x1[ao[2]]-r1*np.sin(theta)
        z1_range = x0[ao[2]]+r0*np.sin(theta),x1[ao[2]]+r1*np.sin(theta)

        z0 = np.clip(np.ceil(np.min([z0_range+z1_range])).astype('int'),0,dims[ao[2]])
        z1 = np.clip(np.floor(np.max([z0_range+z1_range])).astype('int'),0,dims[ao[2]])
        if z1-z0==0:
            return
        
        r = np.linspace(r0_lwr,r1_lwr,z1-z0)
        r_feather = np.linspace(r0_upr,r1_upr,z1-z0)
        feather_fill = np.linspace(r0-r0_lwr,r1-r1_lwr,z1-z0)
        
        #if C[ao[2]]==0.:
        #    import pdb
        #    pdb.set_trace()

        for i,zp in enumerate(range(z0,z1)):
            lam = - (x0[ao[2]] - zp)/C[ao[2]]
            #print(i,zp,length,lam,x0[ao[2]],C[ao[2]])
            P = x0 + C * lam

            minor_axis = r[i]
            major_axis = r[i] / np.cos(theta)
            minor_axis_feather = r_feather[i]
            major_axis_feather = r_feather[i] / np.cos(theta)
            
            #Test for crossing start cap
            crossing_start_cap = False
            crossing_end_cap = False
            if zp>=z0_range[0] and zp<z1_range[0]:
                crossing_start_cap = True
            if zp>=z0_range[1] and zp<z1_range[1]:
                crossing_end_cap = True
                

            xpf, ypf = draw.ellipse(P[ao[0]], P[ao[1]], major_axis_feather, minor_axis_feather, shape=shape, rotation=alpha)
            inside = []
            for x,y in zip(xpf,ypf):
                if orient==0:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([zp,x,y],dtype='float')))
                elif orient==1:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([x,zp,y],dtype='float')))
                else:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([x,y,zp],dtype='float')))
            inside = arr(inside)
            if len(inside)>0:
                xpf,ypf = xpf[inside],ypf[inside]   

            if len(xpf)>0 and len(ypf)>0:
                if orient==0:
                    data[zp,xpf,ypf] = np.clip(data[zp,xpf,ypf] + feather_fill[i],0,1)
                elif orient==1:
                    data[xpf,zp,ypf] = np.clip(data[xpf,zp,ypf] + feather_fill[i],0,1)           
                else:
                    data[xpf,ypf,zp] = np.clip(data[xpf,ypf,zp] + feather_fill[i],0,1)        
                    
            
            xp, yp = draw.ellipse(P[ao[0]], P[ao[1]], major_axis, minor_axis, shape=shape, rotation=alpha)
            inside = []
            for x,y in zip(xp,yp):
                if orient==0:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([zp,x,y],dtype='float')))
                elif orient==1:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([x,zp,y],dtype='float')))
                else:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([x,y,zp],dtype='float')))
            inside = arr(inside)
            if len(inside)>0:
                xp,yp = xp[inside],yp[inside]   

            if len(xp)>0 and len(yp)>0:
                if orient==0:
                    data[zp,xp,yp] = fill_val
                elif orient==1:
                    data[xp,zp,yp] = fill_val            
                else:
                    data[xp,yp,zp] = fill_val

        return data             

    def embed_vessel_in_grid(self,tree,seg_id,extent=None,dims=None,r=10.,c=np.asarray([0.,0.,0.]),
                       R=None,rot=[0.,0.,0.],l=10.,inside=1,outside=0,verbose=False,no_copy=True,
                       sphere_connections=False):
        
        """ 
        Embed a cylinder inside a 3D array
        extent: bounding box for matrix (um)
        dims: grid dimensions
        r: radius (um)
        c: centre (um)
        l: length (um)
        inside: pixel value for inside vessel
        outside: pixel value for outside vessel
        """
        
        extent = self.domain
        dims = self.vessel_grid_size
        start_coord = tree.node_coords[tree.segment_start_node_id[seg_id]]
        end_coord = tree.node_coords[tree.segment_end_node_id[seg_id]]
        rs,re = tree.node_diameter[tree.segment_start_node_id[seg_id]] / 2.,tree.node_diameter[tree.segment_end_node_id[seg_id]] / 2.
        
        self._embed_vessel(start_coord,end_coord,rs,re,self.vessel_grid,dims=dims,extent=extent,tree=tree,sphere_connections=sphere_connections,seg_id=seg_id)
        
    def _embed_vessel(self,start_coord,end_coord,rs,re,vessel_grid,dims=None,extent=None,voxel_size=None,node_type=[None,None],tree=None,sphere_connections=False,
                           seg_id=None,clip_at_grid_resolution=True,fill_mode='diameter',graph_embedding=False,clip_vessel_radius=[None,None],ignore_midline=False):
        c = (end_coord + start_coord) / 2.
        
        if dims is None:
            dims = np.asarray(vessel_grid.shape,dtype='int')
        if extent is None:
            extent = self.domain
        if voxel_size is None:
            voxel_size = self.resolution #(extent[:,1]-extent[:,0]) / dims

        # Radius of vessel in pixels
        #dims,extent = arr(dims),arr(extent)
        
        if clip_vessel_radius[0] is not None or clip_vessel_radius[1] is not None:
            rs = np.clip(rs,clip_vessel_radius[0],clip_vessel_radius[1])
        
        r_pix_s_f = rs/voxel_size
        r_pix_s = np.floor(r_pix_s_f).astype('int32')
        r_pix_e_f = re/voxel_size
        r_pix_e = np.floor(r_pix_e_f).astype('int32')
        r_pix_s[r_pix_s<=0] = 1
        r_pix_e[r_pix_e<=0] = 1
        # Length of vessel in pixels
        l = np.linalg.norm(end_coord-start_coord)
        l_pix_f = l*dims/(extent[:,1]-extent[:,0])
        #l_pix_f = np.clip(l_pix_f,1,dims)
        l_pix = np.clip(l_pix_f,1,dims).astype('int32')
        
        start_pix = self.coord_to_pix(start_coord,extent,dims,clip=False)
        end_pix = self.coord_to_pix(end_coord,extent,dims,clip=False)
        # Check vessel is inside grid
        if (np.all(start_pix<0) and np.all(end_pix<0)) or (np.all(start_pix>=dims) and np.all(end_pix>=dims)):
            return
            
        #vessel_grid = self.embed_cylinder(start_pix,end_pix,r_pix_s_f[0],r_pix_e_f[0],data=vessel_grid)
        
        #if np.any(np.isinf(self.vessel_grid)) or np.any(np.isnan(self.vessel_grid)):
        #    import pdb
        #    pdb.set_trace()
        
        # Midline
        start_pix = self.coord_to_pix(start_coord,extent,dims,voxel_size=voxel_size,clip=False)
        end_pix = self.coord_to_pix(end_coord,extent,dims,voxel_size=voxel_size,clip=False)
        n = int(np.max(np.abs(end_pix - start_pix)))*2 #/np.min([r_pix_s_f[0],r_pix_e_f[0]]))
        if n<=0: # No pixels
            vessel_grid = self.embed_sphere_aniso(start_pix,r_pix_s_f[0],data=vessel_grid,fill_mode=fill_mode,resolution=voxel_size)
            return vessel_grid
            
        xl = np.linspace(start_pix,end_pix,n,dtype='int')
        rl = np.linspace(r_pix_s_f[0],r_pix_e_f[0],n,dtype='float')
        radius = np.linspace(rs,re,n)
        
        for i,coord in enumerate(xl):
            if coord[0]<dims[0] and coord[1]<dims[1] and coord[2]<dims[2] and (clip_at_grid_resolution and rl[i]*2>=1) or not clip_at_grid_resolution:
                vessel_grid = self.embed_sphere_aniso(coord,radius[i],data=vessel_grid,fill_mode=fill_mode,resolution=voxel_size)
                
        # Add spheres to connect cylinders
        #if sphere_connections:
        #    vessel_grid = self.embed_sphere(end_pix,r_pix_e_f[0],data=vessel_grid,fill_mode=fill_mode)
        #    vessel_grid = self.embed_sphere(start_pix,r_pix_s_f[0],data=vessel_grid,fill_mode=fill_mode)                

        if self.vessel_midline is not None and not ignore_midline: # and np.all(start_pix>0) and np.all(start_pix<dims) and np.all(end_pix>0) and np.all(end_pix<dims):
                
                for i,coord in enumerate(xl):
                    if coord[0]<dims[0] and coord[1]<dims[1] and coord[2]<dims[2] and ((clip_at_grid_resolution and rl[i]*2>=1) or not clip_at_grid_resolution):
                        self.vessel_midline[coord[0],coord[1],coord[2]] = 1
                        #vessel_grid = self.embed_sphere(coord,rl[i],data=vessel_grid,fill_mode=fill_mode)

                if graph_embedding: # Set pixel value based on node type
                    if ((clip_at_grid_resolution and rl[0]*2>=1) or not clip_at_grid_resolution) and np.all(start_pix<dims):
                        if tree is not None and seg_id is not None:
                            # Branch nodes?         
                            if tree.node_type[tree.segment_start_node_id[seg_id]]==2:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = 2
                            # Tips?
                            if tree.node_type[tree.segment_start_node_id[seg_id]]==3:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = 3
                        else:
                            if node_type[0] is not None:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = node_type[0]

                    if ((clip_at_grid_resolution and rl[-1]*2>=1) or not clip_at_grid_resolution) and np.all(end_pix<dims):
                        if tree is not None and seg_id is not None:
                            # Branch nodes?         
                            if tree.node_type[tree.segment_end_node_id[seg_id]]==2:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = 2
                            # Tips?
                            if tree.node_type[tree.segment_end_node_id[seg_id]]==3:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = 3
                        else:
                            if node_type[1] is not None:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = node_type[1]
                            
        if self.vessel_midline_diameter is not None: # Set midline pixel value based on radius (useful to have - can just set all values to 1 for midline learning)
            for i,xli in enumerate(xl):
                if xli[0]<dims[0] and xli[1]<dims[1] and xli[2]<dims[2]:
                    if clip_at_grid_resolution and rl[i]*2>=1:
                        self.vessel_midline_diameter[xli[0],xli[1],xli[2]] = rl[i]*2
                    elif not clip_at_grid_resolution:   
                        self.vessel_midline_diameter[xli[0],xli[1],xli[2]] = rl[i]*2

        return vessel_grid
            
    def write_vessel_grid(self,dfile,binary=False,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        #dx = self.resolution
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_grid is None:
            return
        if np.any(np.isinf(self.vessel_grid)):
            import pdb
            pdb.set_trace()

        if nifti:
            if binary:
                #grid_cpy = self.vessel_grid.copy().astype('int16')
                #grid_cpy[grid_cpy>0] = 1
                img = nib.Nifti1Image(self.vessel_grid, tr)
            else:
                img = nib.Nifti1Image(self.vessel_grid ,tr)
            print('Writing vessel grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            nib.save(img,dfile)
        else:
            print('Writing vessel grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            np.savez(dfile, image=self.vessel_grid, tr=tr)
            
    def write_diameter_grid(self,dfile,binary=False,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        #dx = self.resolution
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_diameter is None:
            return
        if np.any(np.isinf(self.vessel_diameter)):
            import pdb
            pdb.set_trace()

        if nifti:
            if binary:
                #grid_cpy = self.vessel_diameter.copy().astype('int16')
                #grid_cpy[grid_cpy>0] = 1
                img = nib.Nifti1Image(self.vessel_diameter, tr)
            else:
                img = nib.Nifti1Image(self.vessel_diameter ,tr)
            print('Writing diameter grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            nib.save(img,dfile)
        else:
            print('Writing diameter grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            np.savez(dfile, image=self.vessel_diameter, tr=tr)

    def write_midline_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_midline is None:
            return
            
        if nifti:
            img = nib.Nifti1Image(self.vessel_midline,tr)
            print('Writing midline grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing midline grid: {}'.format(dfile))
            np.savez(dfile, image=self.vessel_midline, tr=tr)
        
    def write_midline_diameter_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_midline_diameter is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.vessel_midline_diameter,tr)
            print('Writing midline diameter grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing midline diameter grid: {}'.format(dfile))
            np.savez(dfile, image=self.vessel_midline_diameter, tr=tr)
            
    def write_surface_grid(self,dfile,nifti=True,tiff=False):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.surface_grid is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.surface_grid,tr)
            print('Writing surface grid: {}'.format(dfile))
            nib.save(img,dfile)
        if tiff:
            from tifffile import imsave
            import tifffile as tiff
            imsave(dfile, self.surface_grid)
        if not nifti and not tiff:
            print('Writing surface grid: {}'.format(dfile))
            np.savez(dfile, image=self.surface_grid, tr=tr)
            
    def write_label_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.label_grid is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.label_grid,tr)
            print('Writing label grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing label grid: {}'.format(dfile))
            np.savez(dfile, image=self.label_grid, tr=tr)
            
def make_dir(dir,overwrite_existing=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite_existing:
        filesToRemove = [os.path.join(dir,f) for f in os.listdir(dir)]
        for f in filesToRemove:
            os.remove(f)              
            
def embed_graph(embedding_resolution=[10.,10.,10.],embedding_dim=[512,512,512],domain=None,store_midline_diameter=True,store_diameter=True,
                path=None,filename=None,overwrite_existing=True,graph_embedding=False,d_dir=None,m_dir=None,m2_dir=None,
                file_type='nii',centre_mode='fixed',centre_point=arr([0.,0.,0.]),rep_index=0,ignore_null_vessels=False,
                radial_centre=None,prefix='',write=True,sg=None,clip_vessel_radius=[None,None],iter_data=None,fill_mode='binary',clip_at_grid_resolution=True
                ):
                
    noise = False

    if iter_data is not None:
        filename = iter_data['filename']
        embedding_resolution = iter_data['res']
        embedding_dim = iter_data['dim']
        domain = iter_data['domain']
        if 'surface_file' in iter_data.keys():
            surface_file = iter_data['surface_file']
        rep_index = iter_data['rep_index']
            
    if file_type=='npz':
        nifti = False
        file_ext = 'npz'
    else:
        nifti = True
        file_ext = 'nii'

    count = 0
    while True:
        if sg is None:
            print('Embedding file: {}, dim:{}, res:{}, path:{}, d_dir:{}'.format(filename,embedding_dim,embedding_resolution,path,d_dir))    
            sg = spatialgraph.SpatialGraph(initialise=True)
            sg.read(join(path,filename))
            print('Graph read')

        node_coords = sg.fields[0]['data']
        edges = sg.fields[1]['data']
        points = sg.fields[3]['data'] 
        radii = sg.fields[4]['data']
        npts = sg.fields[2]['data']
        
        if centre_mode=='random':
            # Pick a node to centre on
            nedge = edges.shape[0]
            nnode = node_coords.shape[0]
            npoints = np.sum(npts)
            nbins = 20
            hist = np.histogram(radii,bins=nbins)
            npch = int(np.floor(npoints/nbins))
            prob = 1. - np.repeat(hist[0]/npoints,npch)
            prob = prob / np.sum(prob)

            centre_point = np.random.choice(np.arange(0,prob.shape[0]),p=prob)
            rng = int(round(embedding_dim/4.))
            embedding_offset = points[centre_point] + np.random.uniform(-rng,rng,3)
            #centre_node = int(np.random.uniform(0,nnode))
            #embedding_offset = node_coords[centre_node]
        else:
            #centre_point = arr([embedding_dim/2.,embedding_dim/2.,embedding_dim/2.])
            domain_centre = (domain[:,1]-domain[:,0])/2.
            embedding_offset = arr([0.,0.,0.]) #centre_point - domain_centre

        if write:
            if d_dir is None:
                d_dir = path.replace('graph','volume')
            if m_dir is None:
                m_dir = path.replace('graph','volume_midline')
            if m2_dir is None:
                m2_dir = path.replace('graph','volume_midline_diameter')            

            file_stub = filename.replace('.am','')
            file_stub = file_stub.replace('.','p')
            dfile = join(d_dir,file_stub+'.{}'.format(file_ext))
            mfile = join(m_dir,file_stub+'.{}'.format(file_ext))
            mfile2 = join(m2_dir,file_stub+'.{}'.format(file_ext))

        embed_vessels = EmbedVessels(domain=domain,ms=embedding_dim,resolution=embedding_resolution,offset=embedding_offset,store_midline_diameter=store_midline_diameter,store_diameter=store_diameter)

        extent = embed_vessels.domain
        dims = embed_vessels.vessel_grid_size
        grid = embed_vessels.vessel_grid
        diameter = embed_vessels.vessel_diameter
        #print('Embedding initialised: {}, {}'.format(filename,grid.shape))
        
        #breakpoint()
        def inside_domain(coord,extent,margin=100.):
            return coord[0]>=(extent[0,0]-margin) and coord[0]<=(extent[0,1]+margin) and coord[1]>=(extent[1,0]-margin) and coord[1]<=(extent[1,1]+margin) and coord[2]>=(extent[2,0]-margin) and coord[2]<=(extent[2,1]+margin) 
                
        # Check any points are in the domain
        if np.any(points.min(axis=0)>domain[:,1]) or np.any(points.max(axis=0)<domain[:,0]):
            print('No vessels in domain - abandoning graph embedding')
            nvessel = 0
            break
            
        embed_count = 0
        for ei,edge in enumerate(edges):
            x0 = node_coords[edge[0]]
            x1 = node_coords[edge[1]]
            p0 = np.sum(npts[:ei])
            p1 = p0 + npts[ei]
            pts = points[p0:p1] # um
            rads = radii[p0:p1]

            for i,pt in enumerate(pts):
                if i>0:
                    xs,xe = pts[i-1],pts[i]
                    # Check if inside subvolume
                    if inside_domain(xs,extent) and inside_domain(xe,extent):
                        embed_vessels._embed_vessel(xs,xe,rads[i-1],rads[i],grid,dims=dims,extent=extent,clip_at_grid_resolution=clip_at_grid_resolution,fill_mode='binary',graph_embedding=graph_embedding,clip_vessel_radius=clip_vessel_radius)
                        if diameter is not None:
                            embed_vessels._embed_vessel(xs,xe,rads[i-1],rads[i],diameter,dims=dims,extent=extent,clip_at_grid_resolution=clip_at_grid_resolution,fill_mode='diameter',graph_embedding=graph_embedding,clip_vessel_radius=clip_vessel_radius,ignore_midline=True)
                        embed_count += 1

        if embed_count>0 or ignore_null_vessels:
            if embed_count==0:
                print('No vessel verts in domain')
            else:
                print(f'{embed_count} vessel pixels embedded')
            break
        else:
            count += 1
            print('Zero pixels embedded. Trying again (it. {} of 10)'.format(count))
            if count>10:
                return

    # END WHILE

    if write:            
        dfile = dfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))
        mfile = mfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))
        mfile2 = mfile2.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))

        embed_vessels.write_vessel_grid(dfile,binary=False,nifti=nifti)
        embed_vessels.write_midline_grid(mfile,nifti=nifti)
        embed_vessels.write_midline_diameter_grid(mfile2,nifti=nifti) 
    
    print('Complete')
    return embed_vessels, sg
    
def embed_surface(embedding_resolution=[10.,10.,10.],embedding_dim=[512,512,512],domain=None,
                path=None,filename=None,overwrite_existing=True,s_dir=None,
                file_type='npz',centre_mode='fixed',centre_point=arr([0.,0.,0.]),surface_file=None,rep_index=0,
                radial_centre=None,prefix='',write=True,meshes=None,embedding_offset=arr([0.,0.,0.]),iter_data=None,noise=False,
                vals=None,sds=None,embed_vessels=None,define_labels=True
                ):

    if iter_data is not None:
        filename = iter_data['filename']
        embedding_resolution = iter_data['res']
        embedding_dim = iter_data['dim']
        domain = iter_data['domain']
        if 'surface_file' in iter_data.keys():
            surface_file = iter_data['surface_file']
        rep_index = iter_data['rep_index']

    if write:            
        if file_type=='npz':
            nifti = False
            file_ext = 'npz'
        else:
            nifti = True
            file_ext = 'nii'

        if s_dir is None:
            s_dir = path.replace('graph','surface')  
        make_dir(s_dir,overwrite_existing=overwrite_existing)
    
    scene = o3d.t.geometry.RaycastingScene()  
    
    if meshes is None:
        meshes = []
        print('Loading meshes')
        for sf in surface_file:
            meshes.append(o3d.io.read_triangle_mesh(sf))
            
    nsurf = len(meshes)                    
    mesh_ids, all_verts = [],[]
    for mesh in meshes:
        mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        mesh_ids.append(mesh_id)                
        all_verts.append(np.asarray(mesh.vertices))
    all_verts = np.concatenate(all_verts)
    
    if embed_vessels is None:    
        embed_vessels = EmbedVessels(domain=domain,ms=embedding_dim,resolution=embedding_resolution,offset=embedding_offset)
    
    if not np.any(all_verts.min(axis=0)>domain[:,1]) and not np.any(all_verts.max(axis=0)<domain[:,0]):
        print('Ray tracing for surface embedding...')
        subsample_factor = 1
        #x = np.linspace(0,(embedding_dim[0]-1),subsample_factor*embedding_dim[0])
        #y = np.linspace(0,(embedding_dim[1]-1),subsample_factor*embedding_dim[1])
        #z = np.linspace(0,(embedding_dim[2]-1),subsample_factor*embedding_dim[2])
        x = np.linspace(domain[0,0],domain[0,1],subsample_factor*embedding_dim[0])
        y = np.linspace(domain[1,0],domain[1,1],subsample_factor*embedding_dim[1])
        z = np.linspace(domain[2,0],domain[2,1],subsample_factor*embedding_dim[2])
        xg,yg,zg = np.meshgrid(x,y,z)
        xg = xg.flatten()
        yg = yg.flatten()
        zg = zg.flatten()
        if radial_centre is None:
            dir_ = np.tile([0,0,-1],(xg.shape[0],1))
        else:
            dir_ = np.reshape(radial_centre,(3,1))-np.vstack([xg,yg,zg])
            dir_ /= np.linalg.norm(dir_,axis=0)
            dir_ = dir_.transpose()
        ray_dirs = np.vstack([xg,yg,zg,dir_[:,0],dir_[:,1],dir_[:,2]]).transpose()
        # Free memory
        xg,yg,zg,dir_ = None,None,None,None

        # Convert to tensors
        rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        intersections = ans['geometry_ids'].numpy().reshape(embedding_dim).astype('int8')
        intersections[intersections==scene.INVALID_ID] = -1
        t_hit = ans['t_hit'].numpy().reshape(embedding_dim)
        
        # Reverse direction
        ray_dirs[:,3:] *= -1
        rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        intersections2 = ans['geometry_ids'].numpy().reshape(embedding_dim).astype('int8')
        intersections2[intersections2==scene.INVALID_ID] = -1
        t_hit2 = ans['t_hit'].numpy().reshape(embedding_dim)
        
        #o3d.visualization.draw_geometries(meshes,mesh_show_wireframe=True,mesh_show_back_face=True) 

        # Possible noise distributions for signal        
        poisson = np.random.poisson
        norm = np.random.normal
        log_norm = np.random.lognormal
            
        extent = embed_vessels.domain
        dims = embed_vessels.vessel_grid_size
        data = np.zeros(dims,dtype='uint8')
        
        if define_labels:
            labels = np.zeros(dims,dtype='uint8')
 
        if vals is None:
            vals = arr([   100.,  90.,    50.,     55.,  30.,   50.,   10.,   90.,    10.,    100.,  100.,  50.,  25.  ]) * 2
            sds = arr([    20.,   20.,    20.,     20.,  20.,   20.,   20.,   20.,    20.,    20.,   20.,   20.,  20.  ])
            
        for i in range(nsurf):
            sinds = np.where((intersections==i) & (intersections!=-1) & (intersections2!=-1))
            if len(sinds[0])>0:
                if noise:
                    data[sinds] = np.clip(norm(vals[i],sds[i],len(sinds[0])),0,255)
                else:
                    data[sinds] = vals[i]
                if define_labels:
                    labels[sinds] = i + 1
        
        surface = np.rot90(data,k=3,axes=(0,1)).astype('uint8')
        labels = np.rot90(labels,k=3,axes=(0,1)).astype('uint8')
            
        embed_vessels.surface_grid = surface
        
        if define_labels:
            embed_vessels.label_grid = labels
        else:
            embed_vessels.label_grid = None
        
        if write:
            sfile = join(s_dir,file_stub+'_surface.{}'.format(file_ext))
            sfile = sfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,nvessel,rep_index,file_ext))
            embed_vessels.write_surface_grid(sfile,nifti=nifti)
    else:
        embed_vessels.surface_grid = None
        embed_vessels.label_grid = None
        print('No surface verts inside domain')

    #if write:    
    #    dfile = join(s_dir,file_stub+'.{}'.format(file_ext))        
    #    dfile = dfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))
    #    embed_vessels.write_surface_grid(dfile,binary=False,nifti=nifti)
    
    print('Complete')
    return embed_vessels, meshes
    
def batch_embed(path, dim=[128,128,128], graph_embedding=True, file_ext='npz',resolution=[1.,2.,4.,8.,16.,32.,64.,128.], domain=None, nrepeats=5,parallel=True,centre_mode='fixed',surface_only=False, domain_range=None):   
    from os import listdir
    from os.path import isfile, join
    graph_path = join(path,'graph')
    vol_per_file = 1
    files = [f for f in listdir(graph_path) if isfile(join(graph_path, f)) and f.endswith('.am') and f!='.am']
    print('{} graph files located in {}'.format(len(files),graph_path))
    
    parallel = True
    overwrite_existing = True
    ignore_null_vessels = True
    
    opath = path
    
    d_dir = join(opath,'volume') #path.replace('graph','volume')
    make_dir(d_dir,overwrite_existing=overwrite_existing)
    m_dir = join(opath,'volume_midline') #path.replace('graph','volume_midline')
    make_dir(m_dir,overwrite_existing=overwrite_existing)
    m2_dir = join(opath,'volume_midline_diameter') #path.replace('graph','volume_midline')
    make_dir(m2_dir,overwrite_existing=overwrite_existing)
    s_dir = join(opath,'surface')  #path.replace('graph','surface')  
    make_dir(s_dir,overwrite_existing=overwrite_existing)
    
    from pymira import spatialgraph
    # Get spatial extent from the first graph and assume it applies to all others
    sg = spatialgraph.SpatialGraph(initialise=True)
    sg.read(join(path,os.path.join(graph_path,files[0])))
    extent = sg.edge_spatial_extent()
    graph_size = arr([e[1]-e[0] for e in extent])
    
    if type(dim)==int:
        dims = []
        for res in resolution:
            vol_size = int(np.min(graph_size/res))
            if vol_size<dim:
                dims.append(vol_size)
            else:
                dims.append(dim)
    else:
        dims = [dim]
            
    iter_data = []
    for f in files:
        if f.endswith('.am'):
            surface_file = f.replace('.am','_surface.npz')
            if not os.path.exists(join(graph_path,surface_file)):
                surface_file = None
            for i,res in enumerate(resolution):
                for j in range(nrepeats):
                    if domain_range is not None:
                        cur_domain = domain.copy()
                        cur_domain[0,:] += np.random.uniform(domain_range[0,0],domain_range[0,1])
                        cur_domain[1,:] += np.random.uniform(domain_range[1,0],domain_range[1,1])
                        cur_domain[2,:] += np.random.uniform(domain_range[2,0],domain_range[2,1])
                    else:
                        cur_domain = domain.copy()
                    iter_data.append({'filename':f,'res':resolution[i],'dim':dims[i],'surface_file':surface_file,'domain':cur_domain,'rep_index':j})

    #embedding_resolution=10.,embedding_dim=512,path=None,filename=None,overwrite_existing=True,iter_data=None
    #print(iter_data)

    if parallel:
        agents = 20
        func = partial(embed_graph,resolution,dim,domain,graph_path,None,overwrite_existing,graph_embedding,d_dir,m_dir,m2_dir,s_dir,file_ext,centre_mode,None,None,0,ignore_null_vessels)
        import multiprocessing
        with multiprocessing.Pool(processes=agents) as pool:
            result = pool.map(func,iter_data)
    else:
        for f in iter_data:
            embed_graph(resolution,dim,domain,graph_path,None,overwrite_existing,graph_embedding,d_dir,m_dir,m2_dir,s_dir,file_ext,centre_mode,None,None,0,ignore_null_vessels,f)               

