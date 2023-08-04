import os
from os import listdir
from os.path import isfile, join
import numpy as np
arr = np.asarray
import pymira.spatialgraph as sp
import open3d as o3d
from functools import partial

import multiprocessing
import nibabel as nib
from PIL import Image

from pymira import spatialgraph
from retinasim.lsystemlib.embedding import embed_graph, embed_surface
from retinasim import geometry
from retinasim.retina_surface import RetinaSurface
from retinasim.eye import Eye

def coord_to_pixel(coords,domain,dims):

    domain_size = domain[:,1] - domain[:,0]
    coord_fr = (coords - domain[:,0]) / domain_size
    inds = np.round(coord_fr * dims).astype('int')
    return inds

def embed(graph=None,eye=None,filename=None,efile=None,g_file=None,s_file=None,v_file=None,c_file=None,l_file=None,d_file=None,ml_file=None,mld_file=None,prefix='',mesh_file_format='ply',surface_dir=None,dsize=6,dim=arr([500,500,1500]),offset=arr([0.,0.,0.]),output_path=None,domain=None,patch_factor=1,theta=0.,phi=0.,chi=0.,write=True,rotation_centre=None):

    """
    Embed graph and surface in a 3D volume
    offset: embedding offset in um. Mainly used to centre the surface in the volume
    """

    # Eye geometry file
    if eye is None:
        eye = Eye()
        if efile is not None:
            eye.load(efile)

    # Embedding domain size (mm)
    if domain is not None:
        pass
    elif dsize==12: # mm
        domain = arr([[-1000.,11000.],[-6000.,6000.],[eye.domain[2,0],eye.domain[2,1]]])
    elif dsize==6: # mm
        domain = arr([[-1000.,5000.],[-3000.,3000.],[eye.domain[2,0],eye.domain[2,1]]])
    elif dsize==3: # mm
        domain = arr([[-1000.,2000.],[-1500.,1500.],[eye.domain[2,0],eye.domain[2,1]]])
    else:
        # Use simulation domain
        rad_fr = 1.
        domain = eye.domain.astype('float')
        
    #centre = np.mean(domain,axis=1)
    if rotation_centre is None:
        rotation_centre = eye.occular_centre
        rotation_centre[2] = 0.

    # Transform eye geometry parameters to embedding space
    embed_geometry = {}
    embed_geometry['domain'] = domain
    embed_geometry['occular_centre'] = eye.occular_centre
    embed_geometry['optic_disc_centre'] = eye.optic_disc_centre

    # Set geometry
    fov = domain[:,1] - domain[:,0]
    embed_domain = domain.copy()
    #embed_domain[2] /= 4.
    embed_domain[0,:] += offset[0]
    embed_domain[1,:] += offset[1]
    embed_domain[2,:] += offset[2]
    resolution = fov/dim
    
    print(f'Embedding resolution: {resolution}')
    
    # pre-load graph
    if graph is not None:
        sg = graph
    else:
        print(f'Loading graph: {filename}')
        sg = spatialgraph.SpatialGraph(initialise=True)
        sg.read(join(path,filename))

    if theta!=0.:
        # ROTATION TEST
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        #ec = eye.occular_centre
        coords = geometry.rotate_points_about_axis(coords,arr([0.,0.,1.]),theta,centre=rotation_centre) # centre=arr([centre[0],centre[1],0.]))
        points = geometry.rotate_points_about_axis(points,arr([0.,0.,1.]),theta,centre=rotation_centre) #centre=arr([centre[0],centre[1],0.]))
        embed_geometry['optic_disc_centre'] = geometry.rotate_points_about_axis(embed_geometry['optic_disc_centre'],arr([0.,0.,1.]),theta,centre=rotation_centre) #arr([centre[0],centre[1],0.]))
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')
    if phi!=0.:   
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        #ec = eye.occular_centre
        coords = geometry.rotate_points_about_axis(coords,arr([1.,0.,0.]),phi,centre=rotation_centre)
        points = geometry.rotate_points_about_axis(points,arr([1.,0.,0.]),phi,centre=rotation_centre)
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')
    if chi!=0.:   
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        #ec = eye.occular_centre
        coords = geometry.rotate_points_about_axis(coords,arr([0.,1.,0.]),chi,centre=rotation_centre)
        points = geometry.rotate_points_about_axis(points,arr([0.,1.,0.]),chi,centre=rotation_centre)
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')
        
    #print('Writing rotated graph...')
    #sg.write(g_file)
    
    # Find surfaces
    layer_names = [ 'ILM', 'RNFL', 'GCL', 'IPL','INL', 'OPL', 'ONL', 'ELM1', 'ELM2', 'PR1', 'PR2', 'CS', 'CS0'  ]
    sfiles = [join(surface_dir,f'{prefix}retina_surface_{l}.{mesh_file_format}') for l in layer_names]

    if not all([os.path.exists(l) for l in sfiles]):
        print('Could not locate all surface files!')
        #breakpoint()
        return

    # Layer grayscale values and variability
    vals = arr([   200.,  180.,   100.,    110.,  60.,  100.,  20.,   180.,   20.,    200.,  200.,  50.,  50.  ])
    sds = arr([    20.,   20.,    20.,     20.,  20.,   20.,   20.,   20.,    20.,    20.,   20.,   20.,  20.  ])
    
    #pre-load meshes
    print('Loading meshes')
    meshes = []
    for sf in sfiles:
        mesh = o3d.io.read_triangle_mesh(sf)
        if theta!=0.:
            R = mesh.get_rotation_matrix_from_xyz((0.,0., np.deg2rad(theta)))
            mesh.rotate(R, center=rotation_centre) #arr([rotation_centre[0],rotation_centre[1],0.]))
        if phi!=0.:
            ec = eye.occular_centre
            R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(phi),0.,0.))
            mesh.rotate(R, center=rotation_centre)
        if chi!=0.:
            ec = eye.occular_centre
            R = mesh.get_rotation_matrix_from_xyz((0.,np.deg2rad(chi),0.))
            mesh.rotate(R, center=rotation_centre)
        meshes.append(mesh)

    # Embed using patch subsampling
    if patch_factor>1: # even or 1 only
        from scipy.ndimage import zoom
        import cv2
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import RegularGridInterpolator
        from skimage.measure import block_reduce
        pe_x,pe_y,pe_z = np.meshgrid( np.linspace(embed_domain[0,0],embed_domain[0,1],patch_factor+1), 
                                      np.linspace(embed_domain[1,0],embed_domain[1,1],patch_factor+1), # reverse!
                                      np.linspace(embed_domain[2,0],embed_domain[2,1],patch_factor+1), indexing='ij' )
        d_x,d_y,d_z = np.meshgrid(    np.linspace(0,dim[0],patch_factor+1,dtype='int'), 
                                      np.linspace(dim[1],0,patch_factor+1,dtype='int'), # reversed!
                                      np.linspace(0,dim[2],patch_factor+1,dtype='int'), indexing='ij' )                                  
        patch_dim = dim #(dim / patch_factor).astype('int')
        surface = np.zeros(dim,dtype='uint8')
        vessels = np.zeros(dim,dtype='float')
        
        parallel = False
        if parallel: # Doesn't seem to work with open3d... :(
            agents = 40
            
            iter_data = []
            count = 0
            for i in range(1,patch_factor):
                for j in range(1,patch_factor+1):
                    for k in range(1,patch_factor+1):
                        print(i,j,k)
                        patch_embed_domain = arr( [ [pe_x[i-1,j-1,k-1],pe_x[i,j,k]], [pe_y[i-1,j-1,k-1],pe_y[i,j,k]], [pe_z[i-1,j-1,k-1],pe_z[i,j,k]] ] ) 
                        iter_data.append( { 'filename':filename,'res':None,'dim':patch_dim,'domain':patch_embed_domain,'surface_file':sfiles,'rep_index':count } )
                        count += 1
                        
            func = partial(embed_graph,resolution,patch_dim,domain,
                            path,filename,True,True,None,None,None,None,
                            'nii','fixed',arr([0.,0.,0.]),sfiles,0,True,
                            eye.occular_centre,f"{i}_{j}_{k}",False,
                            sg,meshes)

            with multiprocessing.Pool(processes=agents) as pool:
                result = pool.map(func,iter_data)
            breakpoint()
        else:               
            for i in range(1,patch_factor+1):
                for j in range(1,patch_factor+1):
                    for k in range(1,patch_factor+1):
                        print(i,j,k)
                        patch_embed_domain = arr( [ [pe_x[i-1,j-1,k-1],pe_x[i,j,k]], [pe_y[i-1,j-1,k-1],pe_y[i,j,k]], [pe_z[i-1,j-1,k-1],pe_z[i,j,k]] ] )
                        
                        patch_fov = patch_embed_domain[:,1] - patch_embed_domain[:,0]
                        patch_resolution = patch_fov/patch_dim
                        print(f'Patch resolution: {patch_resolution}')
                        
                        embed_v,sg = embed_graph(embedding_resolution=None,embedding_dim=patch_dim,domain=patch_embed_domain,clip_vessel_radius=[2.,None],
                                                        ignore_null_vessels=True,radial_centre=eye.occular_centre,write=False,sg=sg)
                                                       
                        if embed_v.vessel_grid is not None:
                            cur_vess = gaussian_filter(embed_v.vessel_grid,patch_factor/2.)
                            cur_vess = block_reduce(cur_vess,block_size=(patch_factor,patch_factor,patch_factor),func=np.mean)
                            vessels[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_vess
                                                        
                        embed_v,meshes = embed_surface(embedding_resolution=None,embedding_dim=patch_dim,domain=patch_embed_domain,
                                                        overwrite_existing=True,file_type='nii',surface_file=sfiles,rep_index=0,
                                                        radial_centre=eye.occular_centre,write=False,meshes=meshes,vals=vals,sds=sds,embed_vessels=embed_v,define_labels=False)
                                                        
                        if embed_v.surface_grid is not None:
                            cur_surf = gaussian_filter(embed_v.surface_grid,patch_factor/2.)
                            cur_surf = block_reduce(cur_surf,block_size=(patch_factor,patch_factor,patch_factor),func=np.mean)
                            surface[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_surf

            embed_v.vessel_grid = vessels
            embed_v.domain = embed_domain
            embed_v.surface_grid = surface
            embed_v.vessel_midline = vessel_midline
            embed_v.vessel_midline_diameter = vessel_midline_diameter

            # Define labels
            embed_v,meshes = embed_surface(embedding_resolution=None,embedding_dim=dim,domain=embed_domain,
                                                        overwrite_existing=True,file_type='nii',surface_file=sfiles,rep_index=0,
                                                        radial_centre=eye.occular_centre,write=False,meshes=meshes,vals=vals,sds=sds,embed_vessels=embed_v,define_labels=True)
    else:            
        embed_v,sg = embed_graph(embedding_resolution=None,embedding_dim=dim,domain=embed_domain,
                                    rep_index=0,ignore_null_vessels=True,radial_centre=eye.occular_centre,write=False,sg=sg,
                                    clip_vessel_radius=[2.,None])
                                    
        embed_v,meshes = embed_surface(embedding_resolution=None,embedding_dim=dim,domain=embed_domain,
                                                        overwrite_existing=True,file_type='nii',surface_file=sfiles,rep_index=0,
                                                        radial_centre=eye.occular_centre,write=False,meshes=meshes,vals=vals,sds=sds,embed_vessels=embed_v,define_labels=True)

    embed_v.vessel_grid = embed_v.vessel_grid.astype('uint8')
    
    if write==True:               
        if output_path is None:
            output_path = surface_dir.replace('surface','embedding')  
        if not os.path.exists(output_path):
            from pathlib import Path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            #os.mkdir(output_path)                 
          
        if s_file is None:              
            s_file = join(output_path,'tiled_surface.nii')
        if v_file is None:
            v_file = join(output_path,'tiled_vessels.nii')
        if d_file is None:
            d_file = join(output_path,'tiled_diameter.nii')
        if c_file is None:
            c_file = join(output_path,'tiled_composite.nii')
        if l_file is None:
            l_file = join(output_path,'tiled_labels.nii')
        if ml_file is None:
            ml_file = join(output_path,'tiled_midline.nii')
        if mld_file is None:
            mld_file = join(output_path,'tiled_midline_diameter.nii')

        embed_v.write_surface_grid(s_file)
        embed_v.write_label_grid(l_file)
        embed_v.write_vessel_grid(v_file)
        embed_v.write_midline_grid(ml_file)
        #embed_v.write_midline_diameter_grid(mld_file)
        embed_v.write_diameter_grid(d_file)  
    
    # Create blended overlay
    # Blend
    surface = embed_v.surface_grid
    vessels = embed_v.vessel_grid
    composite = surface.copy()*0.

    for i in range(vessels.shape[0]):
        Xim = Image.fromarray((vessels[i]).astype('uint8')).convert('RGBA')
        Xim.putalpha(Image.fromarray(((vessels[i])).astype('uint8'),mode='L'))
        Xbg = Image.fromarray(np.uint8(surface[i])).convert('RGBA')
        Xbg.putalpha(255)
        composite[i] = np.asarray(Image.alpha_composite(Xbg, Xim).convert('L'))
        
    composite = composite.astype('uint8')
    embed_v.surface_grid = composite
    
    if write:
        embed_v.write_surface_grid(c_file)
        embed_v.write_surface_grid(c_file.replace('.nii','.tif'),tiff=True,nifti=False)
    
    return embed_v
