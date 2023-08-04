from retinasim.retina_surface import RetinaSurface
from retinasim.eye import Eye
from retinasim.scripts.project_graph_to_surface import project_to_surface
from retinasim import geometry

import os
join = os.path.join
import numpy as np
arr = np.asarray
import pymira.spatialgraph as sp
import open3d as o3d

def create_surface(path=None,ofile=None,efile=None,mesh_file_format='ply',plot=False,regrid=False,vessel_depth=5000.,plot_file=None,simulate_macula=True,simulate_optic_nerve=True,add_simplex_noise=True,project=True,eye=None):

    if path is None:
        return

    # Eye geometry file
    if eye is None:
        if efile is None:
            efile = join(path,'retina_geometry.p')
        eye = Eye()
        if efile is not None:
            eye.load(efile)

    # Use simulation domain
    domain = eye.domain.astype('float')
    centre = np.mean(domain,axis=1)
    eye.occular_centre[0:2] = centre[0:2]

    # Extend domain to accomodate all of network
    domain *= 1.4
    mesh_domain = domain.copy()

    rs = RetinaSurface(domain=mesh_domain,eye=eye)

    grid_res = 200. # um
    dsize = mesh_domain[:,1]-mesh_domain[:,0]
    n = np.max([int(np.ceil(dsize[0]/grid_res)),100])
    nr, nproj, rng_nsd = 25, 101, 2.5 # 50,101,50
    domain_mode,fill_mode = 'circular','rectangular'

    rs.create_retina_surface(plot=plot,nr=nr,nproj=nproj,n=n,rng_nsd=rng_nsd,vessel_depth=vessel_depth,simulate_macula=simulate_macula,simulate_optic_nerve=simulate_optic_nerve,add_simplex_noise=add_simplex_noise,project=project,plot_file=plot_file,ofile=ofile,regrid=regrid,domain_mode=domain_mode,fill_mode=fill_mode)
